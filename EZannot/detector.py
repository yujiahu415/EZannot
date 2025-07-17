import os
import cv2
import json
import torch
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


