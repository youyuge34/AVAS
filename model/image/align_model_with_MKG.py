'''
带MKG loss的model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from efficientnet_pytorch import EfficientNet
from transformers import BertModel, BertTokenizer, BertConfig

logger = logging.getLogger(__name__)

class ALIGN(nn.Module):
    @classmethod
    def from_pretrained(cls, config):
        weight_path = config['model_path']
        if torch.cuda.is_available():
            pretrained_model = torch.load(weight_path)
        else:
            pretrained_model = torch.load(weight_path, map_location="cpu")

        # 不再使用初始化模型
        config['init_bert_weights'] = False
        config['efficientnet_weights_path'] = None

        model = ALIGN.from_config(config)
        model.load_state_dict(pretrained_model['state_dict'], strict=False)
        print('model.load_state_dict success~')

        return model

    @classmethod # 类方法（不需要实例化类就可以被类本身调用）
    def from_config(cls, conf): # cls : 表示没用被实例化的类本身
        import copy
        cv_type = getattr(conf, "efficientnet_type", "efficientnet-b3")
        bert_layers_num = getattr(conf, "bert_layers_num", 4)
        cv_weights = getattr(conf, "efficientnet_weights_path", None)
        bert_weights = getattr(conf, "bert_path", None)
        init_bert_weights = getattr(conf, "init_bert_weights", True)
        hidden_dim = getattr(conf, "hidden_dim", 512)

        print(cv_type, bert_layers_num, cv_weights, bert_weights, init_bert_weights, hidden_dim)
        # efficientnet-b3 4 None ./bert False clip 512 0.3 1.0
        model = ALIGN(cv_type, bert_layers_num, cv_weights, bert_weights, init_bert_weights, hidden_dim)

        return model

    def __init__(
        self, 
        efficientnet_type='efficientnet-b3',
        bert_layers_num=4,
        image_weights_path=None,
        bert_path=None,
        init_bert_weights=False,
        feature_dim=512
    ):
        super(ALIGN, self).__init__()
        # 初始化 efficient net
        
        if image_weights_path is None:
            self.image_model = EfficientNet.from_name(efficientnet_type, num_classes=feature_dim)
        else:
            self.image_model = EfficientNet.from_pretrained(efficientnet_type, weights_path=image_weights_path, 
                num_classes=feature_dim)
        image_feature_dim = feature_dim

        # 初始化bert模型
        # bert-base-chinese 默认配置
        config = BertConfig.from_pretrained(bert_path+"/config.json")
        config.num_hidden_layers = bert_layers_num
        self.tokenizer = BertTokenizer.from_pretrained(bert_path+"/vocab.txt")
        if not init_bert_weights:
            self.text_model = BertModel(config=config)
        else:
            self.text_model = BertModel.from_pretrained(bert_path+"/pytorch_model.bin", config=config)

        text_feature_dim = config.hidden_size

        # 非线性层
        self.logit_scale = nn.Parameter(torch.tensor(1.))

        self.image_hidden_layer = nn.Linear(in_features=image_feature_dim,
            out_features=feature_dim)
        self.text_hidden_layer = nn.Linear(in_features=text_feature_dim,
            out_features=feature_dim)


        # MKG 新参数
        self.relation_count = 3 # has_cate  has_image has_title
        margin = 1.
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')
        self.norm = 1
        self.dim = 100
        self.relations_emb = self._init_relation_emb()
        self.linear_relation = nn.Linear(self.dim, 512)
        # self.linear_words = nn.Linear(768, 128)
        # self.linear_has_img = nn.Linear(512, 128)

    def _init_relation_emb(self):
        relations_emb = nn.Embedding(num_embeddings=self.relation_count + 1,
                                     embedding_dim=self.dim,
                                     padding_idx=self.relation_count)
        uniform_range = 6 / np.sqrt(self.dim)
        relations_emb.weight.data.uniform_(-uniform_range, uniform_range)
        # -1 to avoid nan for OOV vector
        relations_emb.weight.data[:-1, :].div_(relations_emb.weight.data[:-1, :].norm(p=1, dim=1, keepdim=True))
        return relations_emb

    def get_tokenizer(self):
        return self.tokenizer

    def encode_image(self, image):
        # image_batch = image.shape[0]
        image_embeddings = self.image_model(image)
        image_embeddings = self.image_hidden_layer(image_embeddings)
        image_embeddings = F.normalize(image_embeddings, p=2, dim=1)

        return image_embeddings

    def encode_text(self, text):
        input_ids = text['input_ids']
        attention_mask = text['attention_mask']
        # text_batch = input_ids.shape[0]
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_embeddings = outputs.last_hidden_state[:, 0]
        text_embeddings = self.text_hidden_layer(text_embeddings)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

        return text_embeddings

    def forward(self, image, text, mkg_pos_text_head, mkg_pos_relation, mkg_pos_text_tail,mkg_pos_img_tail, mkg_pos_tail_type,
                mkg_neg_text_tail, mkg_neg_img_tail
                ):
        # clip 对比损失函数
        batch_size = image.shape[0]
        image_embeddings = self.encode_image(image)
        text_embeddings = self.encode_text(text)
        # temp = self.logit_scale.exp()
        sims = image_embeddings @ text_embeddings.t() * self.logit_scale
        labels = torch.arange(batch_size, dtype=torch.long, device=image_embeddings.device)
        loss_clip = (F.cross_entropy(sims, labels) + F.cross_entropy(sims.t(), labels)) / 2

        # transE MKG 损失 head + relation ≈ tail
        mkg_pos_text_head = self.encode_text(mkg_pos_text_head)
        mkg_pos_relation = self.relations_emb(mkg_pos_relation)
        mkg_pos_relation = self.linear_relation(mkg_pos_relation)

        mkg_pos_tail_type -= 1  # 因为type值是1，2，所以index要-1
        pos_tail_name = self.encode_text(mkg_pos_text_tail)
        pos_tail_img = self.encode_image(mkg_pos_img_tail)

        concats = torch.cat([pos_tail_name, pos_tail_img], dim=0)
        # concats: 3 x b x 128
        b = torch.arange(concats.shape[1]).type_as(mkg_pos_tail_type)
        # 取选对应位置的vector。之前算的时候是矩阵统一算的，但是其实不同tail type 的 encoder不一样。
        pos_tail_result = concats[mkg_pos_tail_type.squeeze(), b, :]  # b * 128

        pos_heads = F.normalize(mkg_pos_text_head, p=2, dim=1)
        pos_relation = F.normalize(mkg_pos_relation, p=2, dim=1)
        pos_tail_result = F.normalize(pos_tail_result, p=2, dim=1)
        pos_distance = self._vector_distance(pos_heads, pos_relation, pos_tail_result)


        # MKG 负样本
        mkg_neg_text_head = mkg_pos_text_head
        mkg_neg_relation = mkg_pos_relation
        # mkg_neg_relation = self.linear_relation(mkg_neg_relation)

        mkg_neg_tail_type = mkg_pos_tail_type
        neg_tail_name = self.encode_text(mkg_neg_text_tail)
        neg_tail_img = self.encode_image(mkg_neg_img_tail)

        concats = torch.cat([neg_tail_name, neg_tail_img], dim=0)
        # concats: 3 x b x 128
        b = torch.arange(concats.shape[1]).type_as(mkg_neg_tail_type)
        # 取选对应位置的vector。之前算的时候是矩阵统一算的，但是其实不同tail type 的 encoder不一样。
        neg_tail_result = concats[mkg_neg_tail_type.squeeze(), b, :]  # b * 128

        neg_heads = F.normalize(mkg_neg_text_head, p=2, dim=1)
        neg_relation = F.normalize(mkg_neg_relation, p=2, dim=1)
        neg_tail_result = F.normalize(neg_tail_result, p=2, dim=1)
        neg_distance = self._vector_distance(neg_heads, neg_relation, neg_tail_result)

        # MKG loss
        mkg_loss = self.mkg_loss(pos_distance, neg_distance)

        return loss_clip, mkg_loss, pos_distance, neg_distance

    # def predict(self, triplets: torch.LongTensor):
    #     """Calculated dissimilarity score for given triplets.
    #
    #     :param triplets: triplets in Bx3 shape (B - batch, 3 - head, relation and tail)
    #     :return: dissimilarity score for given triplets
    #     """
    #     return self._distance(triplets)

    def mkg_loss(self, positive_distances, negative_distances):
        target = torch.tensor([-1], dtype=torch.long, device=self.device)
        return self.criterion(positive_distances, negative_distances, target)

    def _distance(self, triplets):
        """Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id.
        该方法无法使用，因为tail的type不固定！
        """
        assert triplets.size()[1] == 3
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        return (self.encode_text(heads) + self.linear_relation(self.relations_emb(relations)) - self.entities_emb(tails)).norm(p=self.norm,dim=1)

    def _vector_distance(self, heads, relations, tails):
        """Triplets should have shape Bx3 where dim 3 are head id, relation id, tail id."""
        # assert triplets.size()[1] == 3
        # heads = triplets[:, 0]
        # relations = triplets[:, 1]
        # tails = triplets[:, 2]
        return (heads + relations - tails).norm(
            p=self.norm,
            dim=1)


def build_model(config):
    if hasattr(config, 'model_path'):
        model = ALIGN.from_pretrained(config)
    else:
        model = ALIGN.from_config(config)

    return model

