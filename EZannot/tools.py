import os
import cv2
import json
import random
import shutil
import itertools
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.ndimage import rotate
from PIL import Image
from screeninfo import get_monitors
from EZannot.sam2.build_sam import build_sam2
from EZannot.sam2.sam2_image_predictor import SAM2ImagePredictor
from EZannot import __version__








def read_annotation(annotation_path):

	annotation_files=[]
	information={}
	classnames=[]
	for i in os.listdir(annotation_path):
		if i.endswith('.json'):
			annotation_files.append(os.path.join(annotation_path,i))
	if len(annotation_files)>0:
		for annotation_file in annotation_files:
			if os.path.exists(annotation_file):
				annotation=json.load(open(annotation_file))
				for i in annotation['images']:
					information[i['file_name']]={'polygons':[],'class_names':[]}
				for i in annotation['categories']:
					if i['id']>0:
						classname=i['name']
						if classname not in classnames:
							classnames.append(classname)
				for i in annotation['annotations']:
					image_name=annotation['images'][int(i['image_id'])]['file_name']
					classname=classnames[int(i['category_id'])-1]
					information[image_name]['polygons'].append([(i['segmentation'][0][x],i['segmentation'][0][x+1]) for x in range(0,len(i['segmentation'][0])-1,2)])
					information[image_name]['class_names'].append(classname)

	return information


def mask_to_polygon(mask):

	contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return []

	max_contour=max(contours,key=cv2.contourArea)
	epsilon=0.005*cv2.arcLength(max_contour,True)
	approx=cv2.approxPolyDP(max_contour,epsilon,True)

	return [tuple(point[0]) for point in approx]


def compute_area(polygon):

	n=len(polygon)
	area=0

	for i in range(n):
		x1,y1=polygon[i]
		x2,y2=polygon[(i+1)%n]
		area+=x1*y2-x2*y1

	return abs(area)/2


def compute_bbox(polygon):

	x_coords,y_coords=zip(*polygon)
	x_min=int(min(x_coords))
	y_min=int(min(y_coords))
	x_max=int(max(x_coords))
	y_max=int(max(y_coords))

	return [x_min,y_min,x_max-x_min,y_max-y_min]


def rotate_point(x,y,cx,cy,angle,image_width,image_height):

	angle_rad=np.radians(-angle)
	cos_theta=np.cos(angle_rad)
	sin_theta=np.sin(angle_rad)
	x_shifted,y_shifted=x-cx,y-cy
	new_x=round(cos_theta*x_shifted-sin_theta*y_shifted+cx)
	new_y=round(sin_theta*x_shifted+cos_theta*y_shifted+cy)

	return max(0,min(image_width-1,new_x)),max(0,min(image_height-1,new_y))


def generate_annotation(path_to_images,information,original_path,result_path,aug_methods,color_map):

	remove=[]

	all_methods=['','_rot1','_rot2','_rot3','_rot4','_rot5','_rot6','_blur']
	options=['_rot7','_flph','_flpv','_brih','_bril','_exph','_expl']
	for r in range(1,len(options)+1):
		all_methods.extend([''.join(c) for c in itertools.combinations(options,r)])

	for i in all_methods:
		if 'random rotation' not in aug_methods:
			if 'rot' in i:
				remove.append(i)
		if 'horizontal flipping' not in aug_methods:
			if 'flph' in i:
				remove.append(i)
		if 'vertical flipping' not in aug_methods:
			if 'flpv' in i:
				remove.append(i)
		if 'random brightening' not in aug_methods:
			if 'brih' in i:
				remove.append(i)
			if 'exph' in i:
				remove.append(i)
		if 'random dimming' not in aug_methods:
			if 'bril' in i:
				remove.append(i)
			if 'expl' in i:
				remove.append(i)
		if 'random blurring' not in aug_methods:
			if 'blur' in i:
				remove.append(i)

	methods=list(set(all_methods)-set(remove))

	coco_format={'info':{'year':'','version':'1','description':'EZannot annotations','contributor':'','url':'https://github.com/yujiahu415/EZannot','date_created':''},'licenses':[],'categories':[],'images':[],'annotations':[]}

	for i,class_name in enumerate(sorted(list(color_map.keys()))):
		coco_format['categories'].append({
			'id':i+1,
			'name':class_name,
			'supercategory':'none'})

	image_id=0
	annotation_id=0

	for image_name in information:

		random.shuffle(methods)

		for m in methods:

			image=cv2.imread(os.path.join(path_to_images,image_name))
			image_width=image.shape[1]
			image_height=image.shape[0]

			if 'rot1' in m:
				angle=np.random.uniform(5,45)
			elif 'rot2' in m:
				angle=np.random.uniform(45,85)
			elif 'rot3' in m:
				angle=90.0
			elif 'rot4' in m:
				angle=np.random.uniform(95,135)
			elif 'rot5' in m:
				angle=np.random.uniform(135,175)
			elif 'rot6' in m:
				angle=180.0
			elif 'rot7' in m:
				angle=np.random.uniform(5,175)
			else:
				angle=None

			if 'flphflpv' in m:
				code=-1
			elif 'flph' in m:
				code=1
			elif 'flpv' in m:
				code=0
			else:
				code=None

			if 'brihbril' in m:
				beta=np.random.uniform(0.6,1.6)
			elif 'brih' in m:
				beta=np.random.uniform(1.1,1.6)
			elif 'bril' in m:
				beta=np.random.uniform(0.6,0.9)
			else:
				beta=None

			if 'exphexpl' in m:
				expo=np.random.uniform(-25,25)
			elif 'exph' in m:
				expo=np.random.uniform(10,25)
			elif 'expl' in m:
				expo=np.random.uniform(-25,-10)
			else:
				expo=None

			if 'blur' in m:
				blur=random.choice([1,3,5])
			else:
				blur=None

			if code is not None:
				image=cv2.flip(image,code)

			if beta is not None:
				image=image.astype('float')
				image=image*beta
				image=np.uint8(np.clip(image,0,255))

			if expo is not None:
				image=image.astype('float')
				image+=expo
				image=np.uint8(np.clip(image,0,255))

			if angle is not None:
				image=rotate(image,angle,reshape=False,prefilter=False)

			if blur is not None:
				image=cv2.GaussianBlur(image,(blur,blur),0)

			new_name=image_name.split('.'+image_name.split('.')[-1])[0]+str(m)+'.'+image_name.split('.')[-1]
			cv2.imwrite(os.path.join(result_path,new_name),image)

			coco_format['images'].append({
				'id':image_id,
				'width':image.shape[1],
				'height':image.shape[0],
				'file_name':new_name})

			polygons=information[image_name]['polygons']

			if len(polygons)>0:

				for j,polygon in enumerate(information[image_name]['polygons']):

					category_id=sorted(list(color_map.keys())).index(information[image_name]['class_names'][j])+1
					segmentation=[np.array(polygon).flatten().tolist()]
					area=compute_area(polygon)
					bbox=compute_bbox(polygon)

					if code is not None or angle is not None:

						new_segmentation=[]
						for seg in segmentation:
							transformed_seg=[]
							for i in range(0,len(seg),2):
								if code==1:
									x=image_width-seg[i]
									y=seg[i+1]
								elif code==0:
									x=seg[i]
									y=image_height-seg[i+1]
								elif code==-1:
									x=image_width-seg[i]
									y=image_height-seg[i+1]
								else:
									x=seg[i]
									y=seg[i+1]
								if angle is not None:
									x,y=rotate_point(x,y,image_width/2,image_height/2,angle,image_width,image_height)
								transformed_seg.extend([int(x),int(y)])
							new_segmentation.append(transformed_seg)
						segmentation=new_segmentation

						[x,y,w,h]=bbox
						if code==1:
							x=image_width-(x+w)
						elif code==0:
							y=image_height-(y+h)
						elif code==-1:
							x=image_width-(x+w)
							y=image_height-(y+h)
						if angle is not None:
							box_points=np.array([[x,y],[x+w,y],[x,y+h],[x+w,y+h]])
							rotated_box=np.array([rotate_point(px,py,image_width/2,image_height/2,angle,image_width,image_height) for px,py in box_points])
							x,y=rotated_box.min(axis=0)
							x_max,y_max=rotated_box.max(axis=0)
							w=x_max-x
							h=y_max-y
						bbox=[int(x),int(y),int(w),int(h)]

					coco_format['annotations'].append({
						'id':annotation_id,
						'image_id':image_id,
						'category_id':category_id,
						'segmentation':segmentation,
						'area':area,
						'bbox':bbox,
						'iscrowd':0
						})

					annotation_id+=1

			image_id+=1

	with open(os.path.join(original_path,'annotations.json'),'w') as json_file:
		json.dump(coco_format,json_file)


def measure_annotation(path_to_images,result_path,information,color_map,show_ids=False,threshold=None):

	data={}
	parameters=['center','area','height','width','perimeter','roundness','intensity']

	for image_name in information:

		filename=os.path.splitext(image_name)[0]
		data[filename]={}
		for object_name in color_map:
			data[filename][object_name]={}
			for parameter in parameters:
				data[filename][object_name][parameter]=[]

		image=cv2.imread(os.path.join(path_to_images,image_name))
		image_width=image.shape[1]
		image_height=image.shape[0]
		thickness=max(1,round(max(image_width,image_height)/960))
		to_annotate=image

		polygons=information[image_name]['polygons']

		if len(polygons)>0:

			for j,polygon in enumerate(information[image_name]['polygons']):

				mask=np.zeros((image_height,image_width),dtype=np.uint8)
				object_name=information[image_name]['class_names'][j]
				pts=np.array(polygon,dtype=np.int32).reshape((-1,1,2))
				cv2.fillPoly(mask,[pts],color=1)
				if threshold is not None:
					excluded_pixels=np.all(image>threshold,axis=2)
					mask[excluded_pixels]=0
				cnts,_=cv2.findContours((mask*255).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
				if len(cnts)>0:
					cnt=sorted(cnts,key=cv2.contourArea,reverse=True)[0]
					area=np.sum(np.array(mask),axis=(0,1))
					perimeter=cv2.arcLength(cnt,closed=True)
					roundness=(perimeter*perimeter)/(4*np.pi*area)
					(_,_),(wd,ht),_=cv2.minAreaRect(cnt)
					intensity=(np.sum(image*cv2.cvtColor(mask*255,cv2.COLOR_GRAY2BGR))/3)/max(area,1)
					if area>0:
						cx=int(cv2.moments(cnt)['m10']/cv2.moments(cnt)['m00'])
						cy=int(cv2.moments(cnt)['m01']/cv2.moments(cnt)['m00'])
						data[filename][object_name]['center'].append((cx,cy))
						data[filename][object_name]['area'].append(area)
						data[filename][object_name]['height'].append(ht)
						data[filename][object_name]['width'].append(wd)
						data[filename][object_name]['perimeter'].append(perimeter)
						data[filename][object_name]['roundness'].append(roundness)
						data[filename][object_name]['intensity'].append(intensity)
						color=(color_map[object_name][2],color_map[object_name][1],color_map[object_name][0])
						if threshold is None:
							cv2.drawContours(to_annotate,[cnt],0,color,thickness)
						else:
							cv2.drawContours(to_annotate,sorted(cnts,key=cv2.contourArea,reverse=True)[:min(2,len(cnts))],-1,color,thickness)
						if show_ids:
							cv2.putText(to_annotate,str(len(data[filename][object_name]['center'])),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,thickness,color,thickness)

		cv2.imwrite(os.path.join(result_path,filename+'_annotated.jpg'),to_annotate)

	with pd.ExcelWriter(os.path.join(result_path,'measurements.xlsx'),engine='openpyxl') as writer:

		for object_name in color_map:

			rows=[]
			columns=['filename','ID']+parameters

			for name,name_data in data.items():
				if object_name in name_data:
					values=zip(*[name_data[object_name][parameter] for parameter in parameters])
					for idx,value in enumerate(values):
						rows.append([name,idx+1]+list(value))

			df=pd.DataFrame(rows,columns=columns)
			df.to_excel(writer,sheet_name=object_name,float_format='%.2f',index=False)

	print('Measurements exported successfully.')


def bbox_intersects(bbox,tile_x,tile_y,tile_w,tile_h):

	x,y,w,h=bbox
	return not (x+w<tile_x or x>tile_x+tile_w or y+h<tile_y or y>tile_y+tile_h)