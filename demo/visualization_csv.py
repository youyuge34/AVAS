import time
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
import os, random, sys
from xml.etree import ElementTree as ET

def main():
    lines = []
    with open(csv_file) as f:
        for line in f:
            line = line.strip()
            if len(line) < 2:
                continue
            path, score = line.split(',')
            lines.append([path, float(score)])
    lines.sort(key= lambda x:-x[1])

    html = ET.Element('html')
    body = ET.Element('body')
    html.append(body)
    table = ET.Element('table', attrib={'border': '2', 'width': '100%'})
    body.append(table)
    print(len(lines))

    iterations = len(lines) // PIC_LIMIT + 1

    for i in tqdm(range(iterations)):
        lines_temp = lines[i*PIC_LIMIT:i*PIC_LIMIT+PIC_LIMIT]
        if len(lines_temp) == 0:
            continue
        tr = ET.Element('tr')
        for j in range(len(lines_temp)):

            td_ = ET.Element('td')
            tr.append(td_)
            td = ET.Element('div', attrib={'style': 'height: 96px; overflow-y:scroll'})
            td_.append(td)
            img = ET.Element('img', attrib={'class': 'imagecontainer', 'width': '96px',
                                            'src': lines_temp[j][0]})
            td.append(img)
            # table.append(tr)

            # tr = ET.Element('tr')
            td = ET.Element('td')
            tr.append(td)
            td.text = str(lines_temp[j][1])
        table.append(tr)

    with open(OUTPUT_NAME, 'wb') as f:
        ET.ElementTree(html).write(f)


if __name__ == '__main__':
    csv_file = 'align_普陀山图片_with_MKG.csv'
    # N_CLUSTER = 8
    PIC_LIMIT = 6
    os.makedirs('html_align_with_mkg', exist_ok=True)
    HTML_NAME = csv_file.split('.')[0] + '_{}.html'.format(
        time.strftime("%H-%M-%S", time.localtime()))
    OUTPUT_NAME = os.path.join('html_align_with_mkg/', HTML_NAME)
    main()