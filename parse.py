import argparse
import json

def parse(in_path, out_path, conf_thres):
	with open(in_path, 'r') as file:
		json_dict = json.load(file)

	# sorted json dictionary by image_id
	json_dict = sorted(json_dict, key=lambda x: x['image_id'])

	output = []
	current = -1
	for box in json_dict:
	    if current != box['image_id'] - 1:
	        for i in range(box['image_id'] - 1 - cur):
	            output.append({'bbox':[], 'score':[], 'label':[]})
	        current = box['image_id'] - 1
	    if float(box['score']) < conf_thres:
	        continue
	    output[current]['bbox'].append((box['bbox'][1], box['bbox'][0], 
	                                box['bbox'][1]+box['bbox'][3], box['bbox'][0]+box['bbox'][2]))
	    output[current]['score'].append(box['score'])
	    output[current]['label'].append(box['category_id'])

	with open(out_path, 'w') as file:
	    json.dump(output, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', type=str, default='./', help='input path')
    parser.add_argument('--out-path', type=str, default='./output.json', help='output path')
    parser.add_argument('--conf-thres', type=float, default=0.0, help='confidence threshold for predicted boxes')
    opt = parser.parse_args()
    
    parse(opt.in_path, opt.out_path, conf_thres)
