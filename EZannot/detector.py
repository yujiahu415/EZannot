import os
import cv2
import json
import torch
from Cellan.detectron2 import model_zoo
from Cellan.detectron2.checkpoint import DetectionCheckpointer
from Cellan.detectron2.config import get_cfg
from Cellan.detectron2.data import MetadataCatalog,DatasetCatalog,build_detection_test_loader
from Cellan.detectron2.data.datasets import register_coco_instances
from Cellan.detectron2.engine import DefaultTrainer,DefaultPredictor
from Cellan.detectron2.utils.visualizer import Visualizer
from Cellan.detectron2.evaluation import COCOEvaluator,inference_on_dataset
from Cellan.detectron2.modeling import build_model



class Detector():

	def __init__(self):

		self.device='cuda' if torch.cuda.is_available() else 'cpu' # whether the GPU is available, if so, use GPU
		self.cell_mapping=None # the celln categories and names in a Detector
		self.inferencing_framesize=None
		self.black_background=None
		self.current_detector=None # the current Detector used for inference


	def train(self,path_to_annotation,path_to_trainingimages,path_to_detector,iteration_num,inference_size,num_rois,black_background=0):

		# path_to_annotation: the path to the .json file that stores the annotations in coco format
		# path_to_trainingimages: the folder that stores all the training images
		# iteration_num: the number of training iterations
		# inference_size: the Detector inferencing frame size
		# num_rois: the batch size of ROI heads per image
		# black_background: whether the background of images to analyze is black/darker

		if str('Cellan_detector_train') in DatasetCatalog.list():
			DatasetCatalog.remove('Cellan_detector_train')
			MetadataCatalog.remove('Cellan_detector_train')

		register_coco_instances('Cellan_detector_train',{},path_to_annotation,path_to_trainingimages)

		datasetcat=DatasetCatalog.get('Cellan_detector_train')
		metadatacat=MetadataCatalog.get('Cellan_detector_train')

		classnames=metadatacat.thing_classes

		model_parameters_dict={}
		model_parameters_dict['cell_names']=[]

		annotation_data=json.load(open(path_to_annotation))

		for i in annotation_data['categories']:
			if i['id']>0:
				model_parameters_dict['cell_names'].append(i['name'])

		print('Cell names in annotation file: '+str(model_parameters_dict['cell_names']))

		cfg=get_cfg()
		cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
		cfg.OUTPUT_DIR=path_to_detector
		cfg.DATASETS.TRAIN=('Cellan_detector_train',)
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
		
		model_parameters_dict['cell_mapping']={}
		model_parameters_dict['inferencing_framesize']=int(inference_size)
		model_parameters_dict['black_background']=int(black_background)

		for i in range(len(classnames)):
			model_parameters_dict['cell_mapping'][i]=classnames[i]

		with open(model_parameters,'w') as f:
			f.write(json.dumps(model_parameters_dict))

		predictor=DefaultPredictor(cfg)
		model=predictor.model

		DetectionCheckpointer(model).resume_or_load(os.path.join(cfg.OUTPUT_DIR,'model_final.pth'))
		model.eval()

		config=os.path.join(cfg.OUTPUT_DIR,'config.yaml')

		with open(config,'w') as f:
			f.write(cfg.dump())

		print('Detector training completed!')


	def test(self,path_to_annotation,path_to_testingimages,path_to_detector,output_path):

		# path_to_annotation: the path to the .json file that stores the annotations in coco format
		# path_to_testingimages: the folder that stores all the ground-truth testing images
		# output_path: the folder that stores the testing images with annotations

		if str('Cellan_detector_test') in DatasetCatalog.list():
			DatasetCatalog.remove('Cellan_detector_test')
			MetadataCatalog.remove('Cellan_detector_test')

		register_coco_instances('Cellan_detector_test',{},path_to_annotation,path_to_testingimages)

		datasetcat=DatasetCatalog.get('Cellan_detector_test')
		metadatacat=MetadataCatalog.get('Cellan_detector_test')

		cellmapping=os.path.join(path_to_detector,'model_parameters.txt')

		with open(cellmapping) as f:
			model_parameters=f.read()

		cell_names=json.loads(model_parameters)['cell_names']
		dt_infersize=int(json.loads(model_parameters)['inferencing_framesize'])
		bg=int(json.loads(model_parameters)['black_background'])

		print('The total categories of cells in this Detector: '+str(cell_names))
		print('The inferencing framesize of this Detector: '+str(dt_infersize))
		if bg==0:
			print('The images that can be analyzed by this Detector have black/darker background')
		else:
			print('The images that can be analyzed by this Detector have white/lighter background')

		cfg=get_cfg()
		cfg.set_new_allowed(True)
		cfg.merge_from_file(os.path.join(path_to_detector,'config.yaml'))
		cfg.MODEL.WEIGHTS=os.path.join(path_to_detector,'model_final.pth')
		cfg.MODEL.DEVICE=self.device

		predictor=DefaultPredictor(cfg)

		for d in datasetcat:
			im=cv2.imread(d['file_name'])
			outputs=predictor(im)
			v=Visualizer(im[:,:,::-1],MetadataCatalog.get('Cellan_detector_test'),scale=1.2)
			out=v.draw_instance_predictions(outputs['instances'].to('cpu'))
			cv2.imwrite(os.path.join(output_path,os.path.basename(d['file_name'])),out.get_image()[:,:,::-1])

		evaluator=COCOEvaluator('Cellan_detector_test',cfg,False,output_dir=output_path)
		val_loader=build_detection_test_loader(cfg,'Cellan_detector_test')

		inference_on_dataset(predictor.model,val_loader,evaluator)

		mAP=evaluator._results['bbox']['AP']

		print(f'The mean average precision (mAP) of the Detector is: {mAP:.4f}%.')
		print('Detector testing completed!')


	def load(self,path_to_detector,cell_kinds):

		# cell_kinds: the catgories of cells / objects to be analyzed

		config=os.path.join(path_to_detector,'config.yaml')
		detector_model=os.path.join(path_to_detector,'model_final.pth')
		cellmapping=os.path.join(path_to_detector,'model_parameters.txt')
		with open(cellmapping) as f:
			model_parameters=f.read()
		self.cell_mapping=json.loads(model_parameters)['cell_mapping']
		cell_names=json.loads(model_parameters)['cell_names']
		self.inferencing_framesize=int(json.loads(model_parameters)['inferencing_framesize'])
		bg=int(json.loads(model_parameters)['black_background'])

		print('The total categories of cells in this Detector: '+str(cell_names))
		print('The cells of interest in this Detector: '+str(cell_kinds))
		print('The inferencing framesize of this Detector: '+str(self.inferencing_framesize))
		if bg==0:
			self.black_background=True
			print('The images that can be analyzed by this Detector have black/darker background')
		else:
			self.black_background=False
			print('The images that can be analyzed by this Detector have white/lighter background')

		cfg=get_cfg()
		cfg.set_new_allowed(True)
		cfg.merge_from_file(config)
		cfg.MODEL.DEVICE=self.device
		self.current_detector=build_model(cfg)
		DetectionCheckpointer(self.current_detector).load(detector_model)
		self.current_detector.eval()


	def inference(self,inputs):

		# inputs: images that the current Detector runs on

		with torch.no_grad():
			outputs=self.current_detector(inputs)

		return outputs


