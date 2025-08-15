import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from tifffile import imread,imwrite
from skimage import exposure
from EZannot.detectron2 import model_zoo
from EZannot.detectron2.checkpoint import DetectionCheckpointer
from EZannot.detectron2.config import get_cfg
from EZannot.detectron2.data import MetadataCatalog,DatasetCatalog,build_detection_test_loader
from EZannot.detectron2.data.datasets import register_coco_instances
from EZannot.detectron2.engine import DefaultTrainer,DefaultPredictor
from EZannot.detectron2.utils.visualizer import Visualizer
from EZannot.detectron2.evaluation import COCOEvaluator,inference_on_dataset
from EZannot.detectron2.modeling import build_model



class Annotator():

	def __init__(self):

		self.device='cuda' if torch.cuda.is_available() else 'cpu' # whether the GPU is available, if so, use GPU
		self.object_mapping=None # the objectn categories and names in a annotator
		self.inferencing_framesize=None
		self.black_background=None
		self.current_annotator=None # the current annotator used for inference


	def train(self,path_to_annotation,path_to_trainingimages,path_to_annotator,iteration_num,inference_size,num_rois,black_background=0):

		# path_to_annotation: the path to the .json file that stores the annotations in coco format
		# path_to_trainingimages: the folder that stores all the training images
		# iteration_num: the number of training iterations
		# inference_size: the annotator inferencing frame size
		# num_rois: the batch size of ROI heads per image
		# black_background: whether the background of images to analyze is black/darker

		if str('EZannot_annotator_train') in DatasetCatalog.list():
			DatasetCatalog.remove('EZannot_annotator_train')
			MetadataCatalog.remove('EZannot_annotator_train')

		register_coco_instances('EZannot_annotator_train',{},path_to_annotation,path_to_trainingimages)

		datasetcat=DatasetCatalog.get('EZannot_annotator_train')
		metadatacat=MetadataCatalog.get('EZannot_annotator_train')

		classnames=metadatacat.thing_classes

		model_parameters_dict={}
		model_parameters_dict['object_names']=[]

		annotation_data=json.load(open(path_to_annotation))

		for i in annotation_data['categories']:
			if i['id']>0:
				model_parameters_dict['object_names'].append(i['name'])

		print('Object names in annotation file: '+str(model_parameters_dict['object_names']))

		cfg=get_cfg()
		cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
		cfg.OUTPUT_DIR=path_to_annotator
		cfg.DATASETS.TRAIN=('EZannot_annotator_train',)
		cfg.DATASETS.TEST=()
		cfg.DATALOADER.NUM_WORKERS=4
		cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
		cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE=num_rois
		cfg.MODEL.ROI_HEADS.NUM_CLASSES=int(len(classnames))
		cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.5
		cfg.MODEL.DEVICE=self.device
		cfg.SOLVER.IMS_PER_BATCH=4
		cfg.SOLVER.MAX_ITER=int(iteration_num)
		cfg.SOLVER.BASE_LR=0.001
		cfg.SOLVER.WARMUP_ITERS=int(iteration_num*0.1)
		cfg.SOLVER.STEPS=(int(iteration_num*0.4),int(iteration_num*0.8))
		cfg.SOLVER.GAMMA=0.5
		cfg.SOLVER.CHECKPOINT_PERIOD=100000000000000000
		cfg.INPUT.MIN_SIZE_TEST=int(inference_size)
		cfg.INPUT.MAX_SIZE_TEST=int(inference_size)
		cfg.INPUT.MIN_SIZE_TRAIN=(int(inference_size),)
		cfg.INPUT.MAX_SIZE_TRAIN=int(inference_size)
		os.makedirs(cfg.OUTPUT_DIR)

		trainer=DefaultTrainer(cfg)
		trainer.resume_or_load(False)
		trainer.train()

		model_parameters=os.path.join(cfg.OUTPUT_DIR,'model_parameters.txt')
		
		model_parameters_dict['object_mapping']={}
		model_parameters_dict['inferencing_framesize']=int(inference_size)
		model_parameters_dict['black_background']=int(black_background)

		for i in range(len(classnames)):
			model_parameters_dict['object_mapping'][i]=classnames[i]

		with open(model_parameters,'w') as f:
			f.write(json.dumps(model_parameters_dict))

		predictor=DefaultPredictor(cfg)
		model=predictor.model

		DetectionCheckpointer(model).resume_or_load(os.path.join(cfg.OUTPUT_DIR,'model_final.pth'))
		model.eval()

		config=os.path.join(cfg.OUTPUT_DIR,'config.yaml')

		with open(config,'w') as f:
			f.write(cfg.dump())

		print('Annotator training completed!')


	def test(self,path_to_annotation,path_to_testingimages,path_to_annotator,output_path):

		# path_to_annotation: the path to the .json file that stores the annotations in coco format
		# path_to_testingimages: the folder that stores all the ground-truth testing images
		# output_path: the folder that stores the testing images with annotations

		if str('EZannot_annotator_test') in DatasetCatalog.list():
			DatasetCatalog.remove('EZannot_annotator_test')
			MetadataCatalog.remove('EZannot_annotator_test')

		register_coco_instances('EZannot_annotator_test',{},path_to_annotation,path_to_testingimages)

		datasetcat=DatasetCatalog.get('EZannot_annotator_test')
		metadatacat=MetadataCatalog.get('EZannot_annotator_test')

		objectmapping=os.path.join(path_to_annotator,'model_parameters.txt')

		with open(objectmapping) as f:
			model_parameters=f.read()

		object_names=json.loads(model_parameters)['object_names']
		dt_infersize=int(json.loads(model_parameters)['inferencing_framesize'])
		bg=int(json.loads(model_parameters)['black_background'])

		print('The total categories of objects in this Annotator: '+str(object_names))
		print('The inferencing framesize of this Annotator: '+str(dt_infersize))
		if bg==0:
			print('The images that can be analyzed by this Annotator have black/darker background')
		else:
			print('The images that can be analyzed by this Annotator have white/lighter background')

		cfg=get_cfg()
		cfg.set_new_allowed(True)
		cfg.merge_from_file(os.path.join(path_to_annotator,'config.yaml'))
		cfg.MODEL.WEIGHTS=os.path.join(path_to_annotator,'model_final.pth')
		cfg.MODEL.DEVICE=self.device

		predictor=DefaultPredictor(cfg)

		for d in datasetcat:
			im=cv2.imread(d['file_name'])
			outputs=predictor(im)
			v=Visualizer(im[:,:,::-1],MetadataCatalog.get('EZannot_annotator_test'),scale=1.2)
			out=v.draw_instance_predictions(outputs['instances'].to('cpu'))
			cv2.imwrite(os.path.join(output_path,os.path.basename(d['file_name'])),out.get_image()[:,:,::-1])

		evaluator=COCOEvaluator('EZannot_annotator_test',cfg,False,output_dir=output_path)
		val_loader=build_detection_test_loader(cfg,'EZannot_annotator_test')

		inference_on_dataset(predictor.model,val_loader,evaluator)

		mAP=evaluator._results['bbox']['AP']

		print(f'The mean average precision (mAP) of the Annotator is: {mAP:.4f}%.')
		print('Annotator testing completed!')


	def load(self,path_to_annotator,object_kinds):

		# object_kinds: the catgories of objects / objects to be analyzed

		config=os.path.join(path_to_annotator,'config.yaml')
		annotator_model=os.path.join(path_to_annotator,'model_final.pth')
		objectmapping=os.path.join(path_to_annotator,'model_parameters.txt')
		with open(objectmapping) as f:
			model_parameters=f.read()
		self.object_mapping=json.loads(model_parameters)['object_mapping']
		object_names=json.loads(model_parameters)['object_names']
		self.inferencing_framesize=int(json.loads(model_parameters)['inferencing_framesize'])
		bg=int(json.loads(model_parameters)['black_background'])

		print('The total categories of objects in this annotator: '+str(object_names))
		print('The objects of interest in this annotator: '+str(object_kinds))
		print('The inferencing framesize of this annotator: '+str(self.inferencing_framesize))
		if bg==0:
			self.black_background=True
			print('The images that can be analyzed by this annotator have black/darker background')
		else:
			self.black_background=False
			print('The images that can be analyzed by this annotator have white/lighter background')

		cfg=get_cfg()
		cfg.set_new_allowed(True)
		cfg.merge_from_file(config)
		cfg.MODEL.DEVICE=self.device
		self.current_annotator=build_model(cfg)
		DetectionCheckpointer(self.current_annotator).load(annotator_model)
		self.current_annotator.eval()


	def inference(self,inputs):

		# inputs: images that the current annotator runs on

		with torch.no_grad():
			outputs=self.current_annotator(inputs)

		return outputs



class AutoAnnotation():

	def __init__(self,image_paths,path_to_annotator,object_kinds,names_colors,detection_threshold=None,filters={}):

		self.image_paths=image_paths
		self.information={}
		self.annotation_path=os.path.dirname(self.image_paths[0])


		self.results_path=os.path.join(results_path,os.path.splitext(os.path.basename(self.path_to_file))[0])


		self.annotator=Annotator()
		self.annotator.load(path_to_annotator,object_kinds)
		self.object_kinds=object_kinds
		self.object_mapping=self.annotator.object_mapping
		self.names_colors=names_colors
		self.detection_threshold=detection_threshold
		if self.detection_threshold is None:
			self.detection_threshold={}
			for object_name in self.object_kinds:
				self.detection_threshold[object_name]=0
		self.fov_dim=self.annotator.inferencing_framesize
		self.black_background=self.annotator.black_background
		self.filters=filters
		self.information={}


	def annotate_images(self):

		coco_format={'info':{'year':'','version':'1','description':'EZannot annotations','contributor':'','url':'https://github.com/yujiahu415/EZannot','date_created':''},'licenses':[],'categories':[],'images':[],'annotations':[]}

		for i,object_name in enumerate(self.object_kinds):
			coco_format['categories'].append({
				'id':i+1,
				'name':object_name,
				'supercategory':'none'})

		data={}
		annotation={'segmentations':[],'class_names':[]}
		total_foreground_area=0
		parameters=['center','area','height','width','perimeter','roundness','intensity']

		for object_name in self.object_kinds:
			data[object_name]={}
			data[object_name]['total_object_area']=0
			for parameter in parameters:
				data[object_name][parameter]=[]

		image_name=os.path.basename(self.path_to_file)
		if os.path.splitext(image_name)[1] in ['.jpg','.JPG','.png','.PNG']:
			image=cv2.imread(self.path_to_file)
		else:
			image=imread(self.path_to_file)
			image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
		image_name=os.path.basename(self.path_to_file)
		width=image.shape[1]
		height=image.shape[0]
		num_w=int(width/self.fov_dim)
		if width%self.fov_dim!=0:
			num_w+=1
		num_h=int(height/self.fov_dim)
		if height%self.fov_dim!=0:
			num_h+=1

		to_annotate=np.uint8(exposure.rescale_intensity(image,out_range=(0,255)))
		thickness=max(1,round(self.fov_dim/960))

		for h in range(num_h):

			for w in range(num_w):

				offset=np.array([[[int(w*self.fov_dim),int(h*self.fov_dim)]]])

				analysis_fov=image[int(h*self.fov_dim):min(int((h+1)*self.fov_dim),height),int(w*self.fov_dim):min(int((w+1)*self.fov_dim),width)]
				detect_fov=np.uint8(exposure.rescale_intensity(analysis_fov,out_range=(0,255)))
				if detect_fov.shape[0]<self.fov_dim or detect_fov.shape[1]<self.fov_dim:
					if self.black_background:
						background_analysis=np.zeros((self.fov_dim,self.fov_dim,3),dtype='uint8')
						background_detect=np.zeros((self.fov_dim,self.fov_dim,3),dtype='uint8')
					else:
						background_analysis=np.uint8(np.ones((self.fov_dim,self.fov_dim,3),dtype='uint8')*255)
						background_detect=np.uint8(np.ones((self.fov_dim,self.fov_dim,3),dtype='uint8')*255)
					background_analysis[:detect_fov.shape[0],:detect_fov.shape[1]]=analysis_fov
					background_detect[:detect_fov.shape[0],:detect_fov.shape[1]]=detect_fov
					analysis_fov=background_analysis
					detect_fov=background_detect
				if self.black_background:
					area_noholes=np.count_nonzero(cv2.threshold(cv2.cvtColor(detect_fov,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])
				else:
					area_noholes=detect_fov.shape[0]*detect_fov.shape[1]-np.count_nonzero(cv2.threshold(cv2.cvtColor(detect_fov,cv2.COLOR_BGR2GRAY),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])

				total_foreground_area+=area_noholes

				output=self.annotator.inference([{'image':torch.as_tensor(detect_fov.astype('float32').transpose(2,0,1))}])
				instances=output[0]['instances'].to('cpu')
				masks=instances.pred_masks.numpy().astype(np.uint8)
				classes=instances.pred_classes.numpy()
				classes=[self.object_mapping[str(x)] for x in classes]
				scores=instances.scores.numpy()

				if len(masks)>0:

					for object_name in self.object_kinds:

						hex_color=self.names_colors[object_name].lstrip('#')
						color=tuple(int(hex_color[i:i+2],16) for i in (0,2,4))
						color=color[::-1]

						object_masks=[masks[a] for a,name in enumerate(classes) if name==object_name]
						object_scores=[scores[a] for a,name in enumerate(classes) if name==object_name]

						if len(object_masks)>0:

							mask_area=np.sum(np.array(object_masks),axis=(1,2))
							exclusion_mask=np.zeros(len(object_masks),dtype=bool)
							exclusion_mask[np.where((np.sum(np.logical_and(np.array(object_masks)[:,None],object_masks),axis=(2,3))/mask_area[:,None]>0.8) & (mask_area[:,None]<mask_area[None,:]))[0]]=True
							object_masks=[m for m,exclude in zip(object_masks,exclusion_mask) if not exclude]
							object_scores=[s for s,exclude in zip(object_scores,exclusion_mask) if not exclude]

							if len(object_masks)>0:

									goodmasks=[object_masks[x] for x,score in enumerate(object_scores) if score>=self.detection_threshold[object_name]]

									if len(goodmasks)>0:

										for mask in goodmasks:
											mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
											if self.expansion is not None:
												mask=cv2.dilate(mask,np.ones((5,5),np.uint8),iterations=self.expansion)
											cnts,_=cv2.findContours((mask*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
											if len(cnts)>0:
												cnt=sorted(cnts,key=cv2.contourArea,reverse=True)[0]
												area=np.sum(np.array(mask),axis=(0,1))
												perimeter=cv2.arcLength(cnt,closed=True)
												roundness=(4*np.pi*area)/(perimeter*perimeter)
												(_,_),(wd,ht),_=cv2.minAreaRect(cnt)
												intensity=np.sum(analysis_fov*cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR))/max(area,1)
												cnt=cnt+offset
												segmentation=cnt.flatten().tolist()
												if 'area' in self.filters:
													if area<self.filters['area'][0] or area>self.filters['area'][1]:
														continue
												if 'perimeter' in self.filters:
													if perimeter<self.filters['perimeter'][0] or perimeter>self.filters['perimeter'][1]:
														continue
												if 'roundness' in self.filters:
													if roundness<self.filters['roundness'][0] or roundness>self.filters['roundness'][1]:
														continue
												if 'height' in self.filters:
													if ht<self.filters['height'][0] or ht>self.filters['height'][1]:
														continue
												if 'width' in self.filters:
													if wd<self.filters['width'][0] or wd>self.filters['width'][1]:
														continue
												if area>0:
													cx=int(cv2.moments(cnt)['m10']/cv2.moments(cnt)['m00'])+int(w*self.fov_dim)
													cy=int(cv2.moments(cnt)['m01']/cv2.moments(cnt)['m00'])+int(h*self.fov_dim)
													data[object_name]['center'].append((cx,cy))
													data[object_name]['area'].append(area)
													data[object_name]['height'].append(ht)
													data[object_name]['width'].append(wd)
													data[object_name]['perimeter'].append(perimeter)
													data[object_name]['roundness'].append(roundness)
													data[object_name]['intensity'].append(intensity)
													annotation['segmentations'].append(segmentation)
													annotation['class_names'].append(object_name)
													cv2.drawContours(to_annotate,[cnt],0,color,thickness)
													if self.show_ids:
														cv2.putText(to_annotate,str(len(data[object_name]['center'])),(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,thickness,color,thickness)
													data[object_name]['total_object_area']+=area

		cv2.imwrite(os.path.join(self.results_path,os.path.splitext(image_name)[0]+'_annotated.'+image_name.split('.')[-1]),to_annotate)

		with pd.ExcelWriter(os.path.join(self.results_path,os.path.splitext(image_name)[0]+'_summary.xlsx'),engine='openpyxl') as writer:

			for object_name in self.object_kinds:

				rows=[]
				columns=['filename','ID']+parameters

				if object_name in data:
					values=zip(*[data[object_name][parameter] for parameter in parameters])
					for idx,value in enumerate(values):
						rows.append([os.path.splitext(image_name)[0],idx+1]+list(value))

				df=pd.DataFrame(rows,columns=columns)
				df.to_excel(writer,sheet_name=object_name,float_format='%.2f',index=False)

		with pd.ExcelWriter(os.path.join(self.results_path,os.path.splitext(image_name)[0]+'_arearatio.xlsx'),engine='openpyxl') as writer:

			for object_name in self.object_kinds:

				dfs={}
				dfs['total_area']=total_foreground_area
				dfs[object_name+'_area']=data[object_name]['total_object_area']
				dfs['area_ratio']=data[object_name]['total_object_area']/total_foreground_area
				dfs=pd.DataFrame(dfs,index=['value'])
				dfs.to_excel(writer,sheet_name=object_name,float_format='%.6f')



		annotation_id=0
		imwrite(os.path.join(self.results_path,image_name),cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

		coco_format['images'].append({
			'id':0,
			'width':image.shape[1],
			'height':image.shape[0],
			'file_name':image_name})

		for i,seg in enumerate(annotation['segmentations']):

			category_id=self.object_kinds.index(annotation['class_names'][i])+1
			polygon=[(seg[x],seg[x+1]) for x in range(0,len(seg)-1,2)]

			n=len(polygon)
			area=0
			for i in range(n):
				x1,y1=polygon[i]
				x2,y2=polygon[(i+1)%n]
				area+=x1*y2-x2*y1
			area=abs(area)/2

			x_coords,y_coords=zip(*polygon)
			x_min=int(min(x_coords))
			y_min=int(min(y_coords))
			x_max=int(max(x_coords))
			y_max=int(max(y_coords))
			bbox=[x_min,y_min,x_max-x_min,y_max-y_min]

			coco_format['annotations'].append({
				'id':annotation_id,
				'image_id':0,
				'category_id':category_id,
				'segmentation':[seg],
				'area':area,
				'bbox':bbox,
				'iscrowd':0
				})

			annotation_id+=1

		with open(os.path.join(self.results_path,'annotations.json'),'w') as json_file:
			json.dump(coco_format,json_file)

		print('Analysis completed!')



