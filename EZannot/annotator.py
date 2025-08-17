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
		self.object_mapping=None # the object categories and names in an Annotator
		self.inferencing_framesize=None
		self.black_background=None
		self.current_annotator=None # the current Annotator used for inference


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

		print('The total categories of objects in this Annotator: '+str(object_names))
		print('The objects of interest in this Annotator: '+str(object_kinds))
		print('The inferencing framesize of this Annotator: '+str(self.inferencing_framesize))
		if bg==0:
			self.black_background=True
			print('The images that can be analyzed by this Annotator have black/darker background')
		else:
			self.black_background=False
			print('The images that can be analyzed by this Annotator have white/lighter background')

		cfg=get_cfg()
		cfg.set_new_allowed(True)
		cfg.merge_from_file(config)
		cfg.MODEL.DEVICE=self.device
		self.current_annotator=build_model(cfg)
		DetectionCheckpointer(self.current_annotator).load(annotator_model)
		self.current_annotator.eval()


	def inference(self,inputs):

		# inputs: images that the current Annotator runs on

		with torch.no_grad():
			outputs=self.current_annotator(inputs)

		return outputs



class AutoAnnotation():

	def __init__(self,image_paths,path_to_annotator,object_kinds,detection_threshold=None,filters={}):

		self.image_paths=image_paths
		self.annotation_path=os.path.dirname(self.image_paths[0])
		self.annotator=Annotator()
		self.annotator.load(path_to_annotator,object_kinds)
		self.object_kinds=object_kinds
		self.object_mapping=self.annotator.object_mapping
		self.detection_threshold=detection_threshold
		if self.detection_threshold is None:
			self.detection_threshold={}
			for object_name in self.object_kinds:
				self.detection_threshold[object_name]=0
		self.filters=filters
		self.information={}


	def annotate_images(self):

		coco_format={
		'info':{'year':'','version':'1','description':'EZannot annotations','contributor':'','url':'https://github.com/yujiahu415/EZannot','date_created':''},
		'licenses':[],
		'categories':[],
		'images':[],
		'annotations':[]}

		for i,object_name in enumerate(self.object_kinds):
			coco_format['categories'].append({
				'id':i+1,
				'name':object_name,
				'supercategory':'none'})

		annotation_id=0

		for image_id,image_path in enumerate(self.image_paths):

			image_name=os.path.basename(image_path)
			self.information[image_name]={'segmentations':[],'class_names':[]}

			if os.path.splitext(image_name)[1] in ['.jpg','.JPG','.png','.PNG']:
				image=cv2.imread(image_path)
			else:
				image=imread(image_path)
				image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

			output=self.annotator.inference([{'image':torch.as_tensor(image.astype('float32').transpose(2,0,1))}])
			instances=output[0]['instances'].to('cpu')
			masks=instances.pred_masks.numpy().astype(np.uint8)
			classes=instances.pred_classes.numpy()
			classes=[self.object_mapping[str(x)] for x in classes]
			scores=instances.scores.numpy()

			if len(masks)>0:

				for object_name in self.object_kinds:

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
										cnts,_=cv2.findContours((mask*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
										if len(cnts)>0:
											cnt=sorted(cnts,key=cv2.contourArea,reverse=True)[0]
											area=np.sum(np.array(mask),axis=(0,1))
											perimeter=cv2.arcLength(cnt,closed=True)
											roundness=(perimeter*perimeter)/(4*np.pi*area)
											(_,_),(wd,ht),_=cv2.minAreaRect(cnt)
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
												self.information[image_name]['segmentations'].append(segmentation)
												self.information[image_name]['class_names'].append(object_name)


			coco_format['images'].append({
				'id':image_id,
				'width':image.shape[1],
				'height':image.shape[0],
				'file_name':image_name})

			for i,seg in enumerate(self.information[image_name]['segmentations']):

				category_id=self.object_kinds.index(self.information[image_name]['class_names'][i])+1
				polygon=[(seg[x],seg[x+1]) for x in range(0,len(seg)-1,2)]

				n=len(polygon)
				area=0
				for idx in range(n):
					x1,y1=polygon[idx]
					x2,y2=polygon[(idx+1)%n]
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
					'image_id':image_id,
					'category_id':category_id,
					'segmentation':[seg],
					'area':area,
					'bbox':bbox,
					'iscrowd':0})

				annotation_id+=1

		with open(os.path.join(self.annotation_path,'annotations.json'),'w') as json_file:
			json.dump(coco_format,json_file)

		print('Auto annotation completed!')


