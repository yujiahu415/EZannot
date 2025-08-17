import os
import cv2
import wx
import wx.lib.agw.hyperlink as hl
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
from .annotator import Annotator,AutoAnnotation



the_absolute_current_path=str(Path(__file__).resolve().parent)



class ColorPicker(wx.Dialog):

	def __init__(self,parent,title,name_and_color):

		super(ColorPicker,self).__init__(parent=None,title=title,size=(200,200))

		self.name_and_color=name_and_color
		name=self.name_and_color[0]
		hex_color=self.name_and_color[1].lstrip('#')
		color=tuple(int(hex_color[i:i+2],16) for i in (0,2,4))

		boxsizer=wx.BoxSizer(wx.VERTICAL)

		self.color_picker=wx.ColourPickerCtrl(self,colour=color)

		button=wx.Button(self,wx.ID_OK,label='Apply')

		boxsizer.Add(0,10,0)
		boxsizer.Add(self.color_picker,0,wx.ALL|wx.CENTER,10)
		boxsizer.Add(button,0,wx.ALL|wx.CENTER,10)
		boxsizer.Add(0,10,0)

		self.SetSizer(boxsizer)



class InitialWindow(wx.Frame):

	def __init__(self,title):

		super(InitialWindow,self).__init__(parent=None,title=title,size=(750,450))
		self.dispaly_window()


	def dispaly_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		self.text_welcome=wx.StaticText(panel,label='Welcome to EZannot!',style=wx.ALIGN_CENTER|wx.ST_ELLIPSIZE_END)
		boxsizer.Add(0,60,0)
		boxsizer.Add(self.text_welcome,0,wx.LEFT|wx.RIGHT|wx.EXPAND,5)
		boxsizer.Add(0,60,0)
		self.text_developers=wx.StaticText(panel,
			label='\nDeveloped by Yujia Hu\n',
			style=wx.ALIGN_CENTER|wx.ST_ELLIPSIZE_END)
		boxsizer.Add(self.text_developers,0,wx.LEFT|wx.RIGHT|wx.EXPAND,5)
		boxsizer.Add(0,60,0)
		
		links=wx.BoxSizer(wx.HORIZONTAL)
		homepage=hl.HyperLinkCtrl(panel,0,'Home Page',URL='https://github.com/yujiahu415/EZannot')
		userguide=hl.HyperLinkCtrl(panel,0,'Extended Guide',URL='')
		links.Add(homepage,0,wx.LEFT|wx.EXPAND,10)
		links.Add(userguide,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(links,0,wx.ALIGN_CENTER,50)
		boxsizer.Add(0,50,0)

		module_modules=wx.BoxSizer(wx.HORIZONTAL)
		button_train=wx.Button(panel,label='Training Module',size=(200,40))
		button_train.Bind(wx.EVT_BUTTON,self.window_train)
		wx.Button.SetToolTip(button_train,'You can train and test an Annotator here. Annotators can automatically annotate all the images for you, which saves huge labor. Depending on the annotation precision, you may or may not need to do manual corrections.')
		button_annotate=wx.Button(panel,label='Annotation Module',size=(200,40))
		button_annotate.Bind(wx.EVT_BUTTON,self.window_annotate)
		wx.Button.SetToolTip(button_annotate,'You can use a trained Annotator for automatic annotation. You can also perform AI-assisted semi-manual annotations to get a small set of initial training data for training an Annotator, or correct the annotations done by an Annotator.')
		module_modules.Add(button_train,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_modules.Add(button_annotate,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_modules,0,wx.ALIGN_CENTER,50)
		boxsizer.Add(0,50,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def window_train(self,event):

		WindowLv1_TrainingModule('Training Module')


	def window_annotate(self,event):

		WindowLv1_AnnotationModule('Annotation Module')



class WindowLv1_TrainingModule(wx.Frame):

	def __init__(self,title):

		super(WindowLv1_TrainingModule,self).__init__(parent=None,title=title,size=(500,250))
		self.dispaly_window()


	def dispaly_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)
		boxsizer.Add(0,60,0)

		button_trainannotators=wx.Button(panel,label='Train Annotators',size=(300,40))
		button_trainannotators.Bind(wx.EVT_BUTTON,self.train_annotators)
		wx.Button.SetToolTip(button_trainannotators,'The trained Annotators can be used to automatically annotate all the images for you, which saves huge labor.')
		boxsizer.Add(button_trainannotators,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,5,0)

		button_testannotators=wx.Button(panel,label='Test Annotators',size=(300,40))
		button_testannotators.Bind(wx.EVT_BUTTON,self.test_annotators)
		wx.Button.SetToolTip(button_testannotators,'Test trained Annotators on the annotated ground-truth image dataset (similar to the image dataset used for training a Annotator).')
		boxsizer.Add(button_testannotators,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,50,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def train_annotators(self,event):

		WindowLv2_TrainAnnotators('Train Annotators')


	def test_annotators(self,event):

		WindowLv2_TestAnnotators('Test Annotators')



class WindowLv2_TrainAnnotators(wx.Frame):

	def __init__(self,title):

		super(WindowLv2_TrainAnnotators,self).__init__(parent=None,title=title,size=(1000,300))
		self.path_to_trainingimages=None
		self.path_to_annotation=None
		self.num_rois=128
		self.inference_size=None
		self.black_background=None
		self.iteration_num=5000
		self.annotator_path=os.path.join(the_absolute_current_path,'annotators')
		self.path_to_annotator=None

		self.dispaly_window()


	def dispaly_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_selectimages=wx.BoxSizer(wx.HORIZONTAL)
		button_selectimages=wx.Button(panel,label='Select the folder containing\nall the training images',size=(300,40))
		button_selectimages.Bind(wx.EVT_BUTTON,self.select_images)
		wx.Button.SetToolTip(button_selectimages,'The folder that stores all the training images.')
		self.text_selectimages=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectimages.Add(button_selectimages,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectimages.Add(self.text_selectimages,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_selectimages,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_selectannotation=wx.BoxSizer(wx.HORIZONTAL)
		button_selectannotation=wx.Button(panel,label='Select the *.json\nannotation file',size=(300,40))
		button_selectannotation.Bind(wx.EVT_BUTTON,self.select_annotation)
		wx.Button.SetToolTip(button_selectannotation,'The .json file that stores the annotation for the training images. Should be in “COCO instance segmentation” format.')
		self.text_selectannotation=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectannotation.Add(button_selectannotation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectannotation.Add(self.text_selectannotation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_selectannotation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_background=wx.BoxSizer(wx.HORIZONTAL)
		button_background=wx.Button(panel,label='Specify whether the background is darker\nthan foreground in training images',size=(300,40))
		button_background.Bind(wx.EVT_BUTTON,self.specify_background)
		wx.Button.SetToolTip(button_background,'This helps the trained Annotator to make up the missing regions when annotating images with the fixed field of view.')
		self.text_background=wx.StaticText(panel,label='Not specified.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_background.Add(button_background,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_background.Add(self.text_background,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_background,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_iterations=wx.BoxSizer(wx.HORIZONTAL)
		button_iterations=wx.Button(panel,label='Specify the iteration number\nfor the Annotator training',size=(300,40))
		button_iterations.Bind(wx.EVT_BUTTON,self.input_iterations)
		wx.Button.SetToolTip(button_iterations,'More training iterations typically yield higher accuracy but take longer.')
		self.text_iterations=wx.StaticText(panel,label='Default: 5000.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_iterations.Add(button_iterations,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_iterations.Add(self.text_iterations,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_iterations,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_train=wx.Button(panel,label='Train the Annotator',size=(300,40))
		button_train.Bind(wx.EVT_BUTTON,self.train_annotator)
		wx.Button.SetToolTip(button_train,'English letters, numbers, “_”, or “-” are acceptable for the names but no “@” or “^”.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_train,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_images(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_trainingimages=dialog.GetPath()
			self.text_selectimages.SetLabel('Path to training images: '+self.path_to_trainingimages+'.')
		dialog.Destroy()


	def select_annotation(self,event):

		wildcard='Annotation File (*.json)|*.json'
		dialog=wx.FileDialog(self, 'Select the annotation file (.json)','',wildcard=wildcard,style=wx.FD_OPEN)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_annotation=dialog.GetPath()
			f=open(self.path_to_annotation)
			info=json.load(f)
			classnames=[]
			for i in info['categories']:
				if i['id']>0:
					classnames.append(i['name'])
			self.text_selectannotation.SetLabel('Object categories in annotation file: '+str(classnames)+'.')
		dialog.Destroy()


	def specify_background(self,event):

		dialog=wx.MessageDialog(self,'Is the background in the images black or darker than foreground?','Darker background?',wx.YES_NO|wx.ICON_QUESTION)
		if dialog.ShowModal()==wx.ID_YES:
			self.black_background=0
			self.text_background.SetLabel('The background in images is black/darker.')
		else:
			self.black_background=1
			self.text_background.SetLabel('The background in images is white/lighter.')
		dialog.Destroy()


	def input_iterations(self,event):

		dialog=wx.NumberEntryDialog(self,'Input the iteration number\nfor the Annotator training','Enter a number:','Iterations',5000,1,1000000)
		if dialog.ShowModal()==wx.ID_OK:
			self.iteration_num=int(dialog.GetValue())
			self.text_iterations.SetLabel('Training iteration number: '+str(self.iteration_num)+'.')
		dialog.Destroy()


	def train_annotator(self,event):

		if self.path_to_trainingimages is None or self.path_to_annotation is None or self.black_background is None:

			wx.MessageBox('No training images / annotation file / background in images specified.','Error',wx.OK|wx.ICON_ERROR)

		else:

			object_sizes=['Sparse and large (e.g., a few animals in an enclosure)','Median (e.g., tissue areas formed by group of cells)','Small (e.g. sparsely distributed cells)','Dense and small (e.g., dense nuclei)']
			dialog=wx.SingleChoiceDialog(self,message='How large are the objects to detect\ncompared to the images?',caption='Object size',choices=object_sizes)
			if dialog.ShowModal()==wx.ID_OK:
				object_size=dialog.GetStringSelection()
				if object_size=='Sparse and large (e.g., a few animals in an enclosure)':
					self.num_rois=128
				elif object_size=='Median (e.g., tissue areas formed by group of cells)':
					self.num_rois=256
				elif object_size=='Small (e.g. sparsely distributed cells)':
					self.num_rois=512
				else:
					self.num_rois=1024
			dialog.Destroy()

			images=[i for i in os.listdir(self.path_to_trainingimages) if i.endswith('.jpg') or i.endswith('.png') or i.endswith('.tif') or i.endswith('.tiff')]
			self.inference_size=int(cv2.imread(os.path.join(self.path_to_trainingimages,images[0])).shape[1])

			do_nothing=False
			stop=False
			while stop is False:
				dialog=wx.TextEntryDialog(self,'Enter a name for the Annotator to train','Annotator name')
				if dialog.ShowModal()==wx.ID_OK:
					if dialog.GetValue()!='':
						self.path_to_annotator=os.path.join(self.annotator_path,dialog.GetValue())
						if not os.path.isdir(self.path_to_annotator):
							stop=True
						else:
							wx.MessageBox('The name already exists.','Error',wx.OK|wx.ICON_ERROR)
				else:
					do_nothing=True
					stop=True
				dialog.Destroy()

			if do_nothing is False:
				AT=Annotator()
				AT.train(self.path_to_annotation,self.path_to_trainingimages,self.path_to_annotator,self.iteration_num,self.inference_size,self.num_rois,black_background=self.black_background)



class WindowLv2_TestAnnotators(wx.Frame):

	def __init__(self,title):

		super(WindowLv2_TestAnnotators,self).__init__(parent=None,title=title,size=(1000,300))
		self.path_to_testingimages=None
		self.path_to_annotation=None
		self.annotator_path=os.path.join(the_absolute_current_path,'annotators')
		self.path_to_annotator=None
		self.output_path=None

		self.dispaly_window()


	def dispaly_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_selectannotator=wx.BoxSizer(wx.HORIZONTAL)
		button_selectannotator=wx.Button(panel,label='Select an Annotator\nto test',size=(300,40))
		button_selectannotator.Bind(wx.EVT_BUTTON,self.select_annotator)
		wx.Button.SetToolTip(button_selectannotator,'The names of objects in the testing dataset should match those in the selected Annotator.')
		self.text_selectannotator=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectannotator.Add(button_selectannotator,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectannotator.Add(self.text_selectannotator,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_selectannotator,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_selectimages=wx.BoxSizer(wx.HORIZONTAL)
		button_selectimages=wx.Button(panel,label='Select the folder containing\nall the testing images',size=(300,40))
		button_selectimages.Bind(wx.EVT_BUTTON,self.select_images)
		wx.Button.SetToolTip(button_selectimages,'The folder that stores all the testing images.')
		self.text_selectimages=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectimages.Add(button_selectimages,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectimages.Add(self.text_selectimages,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_selectimages,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_selectannotation=wx.BoxSizer(wx.HORIZONTAL)
		button_selectannotation=wx.Button(panel,label='Select the *.json\nannotation file',size=(300,40))
		button_selectannotation.Bind(wx.EVT_BUTTON,self.select_annotation)
		wx.Button.SetToolTip(button_selectannotation,'The .json file that stores the annotation for the testing images. Should be in “COCO instance segmentation” format.')
		self.text_selectannotation=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectannotation.Add(button_selectannotation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectannotation.Add(self.text_selectannotation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_selectannotation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_selectoutpath=wx.BoxSizer(wx.HORIZONTAL)
		button_selectoutpath=wx.Button(panel,label='Select the folder to\nstore testing results',size=(300,40))
		button_selectoutpath.Bind(wx.EVT_BUTTON,self.select_outpath)
		wx.Button.SetToolTip(button_selectoutpath,'The folder will stores the testing results.')
		self.text_selectoutpath=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_selectoutpath.Add(button_selectoutpath,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_selectoutpath.Add(self.text_selectoutpath,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_selectoutpath,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		testanddelete=wx.BoxSizer(wx.HORIZONTAL)
		button_test=wx.Button(panel,label='Test the Annotator',size=(300,40))
		button_test.Bind(wx.EVT_BUTTON,self.test_annotator)
		wx.Button.SetToolTip(button_test,'Test the selected Annotator on the annotated, ground-truth testing images.')
		button_delete=wx.Button(panel,label='Delete a Annotator',size=(300,40))
		button_delete.Bind(wx.EVT_BUTTON,self.remove_annotator)
		wx.Button.SetToolTip(button_delete,'Permanently delete a Annotator. The deletion CANNOT be restored.')
		testanddelete.Add(button_test,0,wx.RIGHT,50)
		testanddelete.Add(button_delete,0,wx.LEFT,50)
		boxsizer.Add(0,5,0)
		boxsizer.Add(testanddelete,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_annotator(self,event):

		annotators=[i for i in os.listdir(self.annotator_path) if os.path.isdir(os.path.join(self.annotator_path,i))]
		if '__pycache__' in annotators:
			annotators.remove('__pycache__')
		if '__init__' in annotators:
			annotators.remove('__init__')
		if '__init__.py' in annotators:
			annotators.remove('__init__.py')
		annotators.sort()

		dialog=wx.SingleChoiceDialog(self,message='Select a Annotator to test',caption='Test a Annotator',choices=annotators)
		if dialog.ShowModal()==wx.ID_OK:
			annotator=dialog.GetStringSelection()
			self.path_to_annotator=os.path.join(self.annotator_path,annotator)
			objectmapping=os.path.join(self.path_to_annotator,'model_parameters.txt')
			with open(objectmapping) as f:
				model_parameters=f.read()
			object_names=json.loads(model_parameters)['object_names']
			self.text_selectannotator.SetLabel('Selected: '+str(annotator)+' (objects: '+str(object_names)+').')
		dialog.Destroy()


	def select_images(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_testingimages=dialog.GetPath()
			self.text_selectimages.SetLabel('Path to testing images: '+self.path_to_testingimages+'.')
		dialog.Destroy()


	def select_annotation(self,event):

		wildcard='Annotation File (*.json)|*.json'
		dialog=wx.FileDialog(self, 'Select the annotation file (.json)','',wildcard=wildcard,style=wx.FD_OPEN)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_annotation=dialog.GetPath()
			f=open(self.path_to_annotation)
			info=json.load(f)
			classnames=[]
			for i in info['categories']:
				if i['id']>0:
					classnames.append(i['name'])
			self.text_selectannotation.SetLabel('Object categories in annotation file: '+str(classnames)+'.')
		dialog.Destroy()


	def select_outpath(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.output_path=dialog.GetPath()
			self.text_selectoutpath.SetLabel('Path to testing images: '+self.output_path+'.')
		dialog.Destroy()


	def test_annotator(self,event):

		if self.path_to_annotator is None or self.path_to_testingimages is None or self.path_to_annotation is None or self.output_path is None:
			wx.MessageBox('No Annotator / training images / annotation file / output path selected.','Error',wx.OK|wx.ICON_ERROR)
		else:
			AT=Annotator()
			AT.test(self.path_to_annotation,self.path_to_testingimages,self.path_to_annotator,self.output_path)


	def remove_annotator(self,event):

		annotators=[i for i in os.listdir(self.annotator_path) if os.path.isdir(os.path.join(self.annotator_path,i))]
		if '__pycache__' in annotators:
			annotators.remove('__pycache__')
		if '__init__' in annotators:
			annotators.remove('__init__')
		if '__init__.py' in annotators:
			annotators.remove('__init__.py')
		annotators.sort()

		dialog=wx.SingleChoiceDialog(self,message='Select a Annotator to delete',caption='Delete a Annotator',choices=annotators)
		if dialog.ShowModal()==wx.ID_OK:
			annotator=dialog.GetStringSelection()
			dialog1=wx.MessageDialog(self,'Delete '+str(annotator)+'?','CANNOT be restored!',wx.YES_NO|wx.ICON_QUESTION)
			if dialog1.ShowModal()==wx.ID_YES:
				shutil.rmtree(os.path.join(self.annotator_path,annotator))
			dialog1.Destroy()
		dialog.Destroy()



class WindowLv1_AnnotationModule(wx.Frame):

	def __init__(self,title):

		super(WindowLv1_AnnotationModule,self).__init__(parent=None,title=title,size=(500,250))
		self.dispaly_window()


	def dispaly_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)
		boxsizer.Add(0,60,0)

		button_manualannotate=wx.Button(panel,label='Manual annotation',size=(300,40))
		button_manualannotate.Bind(wx.EVT_BUTTON,self.manual_annotate)
		wx.Button.SetToolTip(button_manualannotate,'Use AI assistance to manually annotate a small set of initial training images for training an Annotator or refine the automatic annotations performed by an Annotator.')
		boxsizer.Add(button_manualannotate,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,5,0)

		button_autoannotate=wx.Button(panel,label='Automatic annotation',size=(300,40))
		button_autoannotate.Bind(wx.EVT_BUTTON,self.auto_annotate)
		wx.Button.SetToolTip(button_autoannotate,'Use a trained Annotators to automatically annotate selected images for you.')
		boxsizer.Add(button_autoannotate,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,50,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def manual_annotate(self,event):

		WindowLv2_ManualAnnotate('Manual Annotate')


	def auto_annotate(self,event):

		WindowLv2_AutoAnnotate('Auto Annotate')



class WindowLv2_ManualAnnotate(wx.Frame):

	def __init__(self,title):

		super(WindowLv2_ManualAnnotate,self).__init__(parent=None,title=title,size=(1000,350))
		self.path_to_images=None
		self.result_path=None
		self.model_cp=None
		self.model_cfg=None
		self.color_map={}
		self.aug_methods=[]

		self.display_window()


	def display_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_input=wx.BoxSizer(wx.HORIZONTAL)
		button_input=wx.Button(panel,label='Select the image(s)\nfor annotation',size=(300,40))
		button_input.Bind(wx.EVT_BUTTON,self.select_images)
		wx.Button.SetToolTip(button_input,'Select one or more images. Common image formats (jpg, png, tif) are supported. If there is an annotation file in the same folder, EZannot will read the annotation file and show all the existing annotations.')
		self.text_input=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_input.Add(button_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_input.Add(self.text_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_outputfolder=wx.BoxSizer(wx.HORIZONTAL)
		button_outputfolder=wx.Button(panel,label='Select a folder to store\nthe annotated images',size=(300,40))
		button_outputfolder.Bind(wx.EVT_BUTTON,self.select_outpath)
		wx.Button.SetToolTip(button_outputfolder,'Copies of images (including augmented ones) and the annotation file will be stored in this folder. The annotation file for the original (unaugmented) images will be stored in the origianl image folder.')
		self.text_outputfolder=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_outputfolder.Add(button_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_outputfolder.Add(self.text_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_model=wx.BoxSizer(wx.HORIZONTAL)
		button_model=wx.Button(panel,label='Set up the SAM2 model for\nAI-assisted annotation',size=(300,40))
		button_model.Bind(wx.EVT_BUTTON,self.select_model)
		wx.Button.SetToolTip(button_model,'Choose the SAM2 model. If select from a folder, make sure the folder stores a checkpoint (*.pt) file and a corresponding model config (*.yaml) file.')
		self.text_model=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_model.Add(button_model,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_model.Add(self.text_model,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_model,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_classes=wx.BoxSizer(wx.HORIZONTAL)
		button_classes=wx.Button(panel,label='Specify the object classes and\ntheir annotation colors',size=(300,40))
		button_classes.Bind(wx.EVT_BUTTON,self.specify_classes)
		wx.Button.SetToolTip(button_classes,'Enter the name of each class and specify its annotation color.')
		self.text_classes=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_classes.Add(button_classes,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_classes.Add(self.text_classes,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_classes,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_augmentation=wx.BoxSizer(wx.HORIZONTAL)
		button_augmentation=wx.Button(panel,label='Specify the augmentation methods\nfor the annotated images',size=(300,40))
		button_augmentation.Bind(wx.EVT_BUTTON,self.specify_augmentation)
		wx.Button.SetToolTip(button_augmentation,
			'Augmentation can greatly enhance the training efficiency. But for the first time of annotating an image set, you can skip this to build an unaugmented, origianl annotated image set and import it to EZannot later to perform the augmentation.')
		self.text_augmentation=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_augmentation.Add(button_augmentation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_augmentation.Add(self.text_augmentation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_augmentation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_startannotation=wx.Button(panel,label='Start to annotate images',size=(300,40))
		button_startannotation.Bind(wx.EVT_BUTTON,self.start_annotation)
		wx.Button.SetToolTip(button_startannotation,'Manually annotate objects in images.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_startannotation,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_images(self,event):

		wildcard='Image files(*.jpg;*.jpeg;*.png;*.tif;*.tiff)|*.jpg;*.jpeg;*.png;*.tif;*.tiff'
		dialog=wx.FileDialog(self,'Select images(s)','','',wildcard,style=wx.FD_MULTIPLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_images=dialog.GetPaths()
			self.path_to_images.sort()
			path=os.path.dirname(self.path_to_images[0])
			self.text_input.SetLabel('Select: '+str(len(self.path_to_images))+' images in'+str(path)+'.')
		dialog.Destroy()


	def select_outpath(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.result_path=dialog.GetPath()
			self.text_outputfolder.SetLabel('The annotated images will be in: '+self.result_path+'.')
		dialog.Destroy()


	def select_model(self,event):

		path_to_sam2_model=None
		sam2_model_path=os.path.join(the_absolute_current_path,'sam2 models')
		sam2_models=[i for i in os.listdir(sam2_model_path) if os.path.isdir(os.path.join(sam2_model_path,i))]
		if '__pycache__' in sam2_models:
			sam2_models.remove('__pycache__')
		if '__init__' in sam2_models:
			sam2_models.remove('__init__')
		if '__init__.py' in sam2_models:
			sam2_models.remove('__init__.py')
		sam2_models.sort()
		if 'Choose a new directory of the SAM2 model' not in sam2_models:
			sam2_models.append('Choose a new directory of the SAM2 model')

		dialog=wx.SingleChoiceDialog(self,message='Select a SAM2 model for AI-assisted annotation.',caption='Select a SAM2 model',choices=sam2_models)
		if dialog.ShowModal()==wx.ID_OK:
			sam2_model=dialog.GetStringSelection()
			if sam2_model=='Choose a new directory of the SAM2 model':
				dialog1=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
				if dialog1.ShowModal()==wx.ID_OK:
					path_to_sam2_model=dialog1.GetPath()
				else:
					path_to_sam2_model=None
				dialog1.Destroy()
			else:
				path_to_sam2_model=os.path.join(sam2_model_path,sam2_model)
		dialog.Destroy()

		if path_to_sam2_model is None:
			wx.MessageBox('No SAM2 model is set up. The AI assistance function is OFF.','AI assistance OFF',wx.ICON_INFORMATION)
			self.text_model.SetLabel('No SAM2 model is set up. The AI assistance function is OFF.')
		else:
			for i in os.listdir(path_to_sam2_model):
				if i.endswith('.pt') and i.split('sam')[0]!='._':
					self.model_cp=os.path.join(path_to_sam2_model,i)
				if i.endswith('.yaml') and i.split('sam')[0]!='._':
					self.model_cfg=os.path.join(path_to_sam2_model,i)
			if self.model_cp is None:
				self.text_model.SetLabel('Missing checkpoint file.')
			elif self.model_cfg is None:
				self.text_model.SetLabel('Missing config file.')
			else:
				self.text_model.SetLabel('Checkpoint: '+str(os.path.basename(self.model_cp))+'; Config: '+str(os.path.basename(self.model_cfg))+'.')


	def specify_classes(self,event):

		if self.path_to_images is None:

			wx.MessageBox('No input images(s).','Error',wx.OK|wx.ICON_ERROR)

		else:

			annotation_files=[]
			color_map={}
			classnames=''
			entry=None
			for i in os.listdir(os.path.dirname(self.path_to_images[0])):
				if i.endswith('.json'):
					annotation_files.append(os.path.join(os.path.dirname(self.path_to_images[0]),i))

			if len(annotation_files)>0:
				for annotation_file in annotation_files:
					if os.path.exists(annotation_file):
						annotation=json.load(open(annotation_file))
						for i in annotation['categories']:
							if i['id']>0:
								classnames=classnames+i['name']+','
				classnames=classnames[:-1]
				dialog=wx.MessageDialog(self,'Current classnames are: '+classnames+'.\nDo you want to modify the classnames?','Modify classnames?',wx.YES_NO|wx.ICON_QUESTION)
				if dialog.ShowModal()==wx.ID_YES:
					dialog1=wx.TextEntryDialog(self,'Enter the names of objects to annotate\n(use "," to separate each name)','Object class names',value=classnames)
					if dialog1.ShowModal()==wx.ID_OK:
						entry=dialog1.GetValue()
					dialog1.Destroy()
				else:
					entry=classnames
				dialog.Destroy()
			else:
				dialog=wx.TextEntryDialog(self,'Enter the names of objects to annotate\n(use "," to separate each name)','Object class names')
				if dialog.ShowModal()==wx.ID_OK:
					entry=dialog.GetValue()
				dialog.Destroy()

			if entry:
				try:
					for i in entry.split(','):
						color_map[i]='#%02x%02x%02x'%(random.randint(0,255),random.randint(0,255),random.randint(0,255))
				except:
					color_map={}
					wx.MessageBox('Please enter the object class names in\ncorrect format! For example: apple,orange,pear','Error',wx.OK|wx.ICON_ERROR)

			if len(color_map)>0:
				for classname in color_map:
					dialog=ColorPicker(self,f'Color for annotating {classname}',[classname,color_map[classname]])
					if dialog.ShowModal()==wx.ID_OK:
						(r,b,g,_)=dialog.color_picker.GetColour()
						self.color_map[classname]=(r,b,g)
					dialog.Destroy()
				self.text_classes.SetLabel('Classname:color: '+str(self.color_map)+'.')
			else:
				self.text_classes.SetLabel('None.')


	def specify_augmentation(self,event):

		aug_methods=['random rotation','horizontal flipping','vertical flipping','random brightening','random dimming','random blurring']
		selected=''
		dialog=wx.MultiChoiceDialog(self,message='Data augmentation methods',caption='Augmentation methods',choices=aug_methods)
		if dialog.ShowModal()==wx.ID_OK:
			self.aug_methods=[aug_methods[i] for i in dialog.GetSelections()]
			for i in self.aug_methods:
				if selected=='':
					selected=selected+i
				else:
					selected=selected+','+i
		else:
			self.aug_methods=[]
			selected='none'
		dialog.Destroy()

		if len(self.aug_methods)<=0:
			selected='none'

		self.text_augmentation.SetLabel('Augmentation methods: '+selected+'.')	


	def start_annotation(self,event):

		if self.path_to_images is None or self.result_path is None or len(self.color_map)==0:
			wx.MessageBox('No input images(s) / output folder / class names.','Error',wx.OK|wx.ICON_ERROR)
		else:
			WindowLv3_AnnotateImages(None,'Manually Annotate Images',self.path_to_images,self.result_path,self.color_map,self.aug_methods,model_cp=self.model_cp,model_cfg=self.model_cfg)



class WindowLv3_AnnotateImages(wx.Frame):

	def __init__(self,parent,title,path_to_images,result_path,color_map,aug_methods,model_cp=None,model_cfg=None):

		monitor=get_monitors()[0]
		monitor_w,monitor_h=monitor.width,monitor.height

		super().__init__(parent,title=title,pos=(10,0),size=(get_monitors()[0].width-20,get_monitors()[0].height-50))

		self.image_paths=path_to_images
		self.result_path=result_path
		self.color_map=color_map
		self.aug_methods=aug_methods
		self.model_cp=model_cp
		self.model_cfg=model_cfg
		self.current_image_id=0
		self.current_image=None
		self.current_segmentation=None
		self.current_polygon=[]
		self.current_classname=list(self.color_map.keys())[0]
		self.information={}
		self.foreground_points=[]
		self.background_points=[]
		self.selected_point=None
		self.start_modify=False
		self.show_name=False
		self.AI_help=False

		annotation_files=[]
		for i in os.listdir(os.path.dirname(self.image_paths[0])):
			if i.endswith('.json'):
				annotation_files.append(os.path.join(os.path.dirname(self.image_paths[0]),i))
		if len(annotation_files)>0:
			for annotation_file in annotation_files:
				if os.path.exists(annotation_file):
					annotation=json.load(open(annotation_file))
					for i in annotation['images']:
						self.information[i['file_name']]={'polygons':[],'class_names':[]}
					for i in annotation['annotations']:
						image_name=annotation['images'][int(i['image_id'])]['file_name']
						classname=list(self.color_map.keys())[int(i['category_id'])-1]
						self.information[image_name]['polygons'].append([(i['segmentation'][0][x],i['segmentation'][0][x+1]) for x in range(0,len(i['segmentation'][0])-1,2)])
						self.information[image_name]['class_names'].append(classname)

		self.init_ui()
		self.load_current_image()


	def sam2_model(self):

		device='cuda' if torch.cuda.is_available() else 'cpu'
		predictor=SAM2ImagePredictor(build_sam2(self.model_cfg,self.model_cp,device=device))
		return predictor


	def init_ui(self):

		panel=wx.Panel(self)
		vbox=wx.BoxSizer(wx.VERTICAL)
		hbox=wx.BoxSizer(wx.HORIZONTAL)

		self.ai_button=wx.ToggleButton(panel,label='AI Help: OFF',size=(150,30))
		self.ai_button.Bind(wx.EVT_TOGGLEBUTTON,self.toggle_ai)
		hbox.Add(self.ai_button,flag=wx.ALL,border=2)

		self.prev_button=wx.Button(panel,label='← Prev',size=(150,30))
		self.prev_button.Bind(wx.EVT_BUTTON,self.previous_image)
		hbox.Add(self.prev_button,flag=wx.ALL,border=2)

		self.next_button=wx.Button(panel,label='Next →',size=(150,30))
		self.next_button.Bind(wx.EVT_BUTTON,self.next_image)
		hbox.Add(self.next_button,flag=wx.ALL,border=2)

		self.delete_button=wx.Button(panel,label='Delete',size=(150,30))
		self.delete_button.Bind(wx.EVT_BUTTON,self.delete_image)
		hbox.Add(self.delete_button,flag=wx.ALL,border=2)

		self.export_button=wx.Button(panel,label='Export Annotations',size=(150,30))
		self.export_button.Bind(wx.EVT_BUTTON,self.export_annotations)
		hbox.Add(self.export_button,flag=wx.ALL,border=2)

		self.measure_button=wx.Button(panel,label='Measure Annotations',size=(150,30))
		self.measure_button.Bind(wx.EVT_BUTTON,self.measure_annotations)
		hbox.Add(self.measure_button,flag=wx.ALL,border=2)
		vbox.Add(hbox,flag=wx.ALIGN_CENTER|wx.TOP,border=5)

		self.scrolled_canvas=wx.ScrolledWindow(panel,style=wx.VSCROLL|wx.HSCROLL)
		self.scrolled_canvas.SetScrollRate(10,10)
		self.canvas=wx.Panel(self.scrolled_canvas,pos=(10,0),size=(get_monitors()[0].width-20,get_monitors()[0].height-50))
		self.canvas.SetBackgroundColour('black')

		self.canvas.Bind(wx.EVT_PAINT,self.on_paint)
		self.canvas.Bind(wx.EVT_LEFT_DOWN,self.on_left_click)
		self.canvas.Bind(wx.EVT_RIGHT_DOWN,self.on_right_click)
		self.canvas.Bind(wx.EVT_MOTION,self.on_left_move)
		self.canvas.Bind(wx.EVT_LEFT_UP,self.on_left_up)

		self.scrolled_canvas.SetSizer(wx.BoxSizer(wx.VERTICAL))
		self.scrolled_canvas.GetSizer().Add(self.canvas,proportion=1,flag=wx.EXPAND|wx.ALL,border=5)
		vbox.Add(self.scrolled_canvas,proportion=1,flag=wx.EXPAND|wx.ALL,border=5)

		panel.SetSizer(vbox)
		self.Bind(wx.EVT_CHAR_HOOK,self.on_key_press)
		self.Show()


	def toggle_ai(self,event):

		if self.model_cp is None or self.model_cfg is None:

			self.ai_button.SetLabel('AI Help: OFF')
			wx.MessageBox('SAM2 model has not been set up.','Error',wx.ICON_ERROR)

		else:

			self.AI_help=self.ai_button.GetValue()
			self.ai_button.SetLabel(f'AI Help: {"ON" if self.AI_help else "OFF"}')

			if self.AI_help:
				image=Image.open(self.image_paths[self.current_image_id])
				image=np.array(image.convert('RGB'))
				self.sam2=self.sam2_model()
				self.sam2.set_image(image)

		self.canvas.SetFocus()


	def load_current_image(self):

		if self.image_paths:
			path=self.image_paths[self.current_image_id]
			self.current_image=wx.Image(path,wx.BITMAP_TYPE_ANY)
			img_width,img_height=self.current_image.GetSize()
			self.scrolled_canvas.SetVirtualSize((img_width,img_height))
			self.canvas.SetSize((img_width,img_height))
			self.scrolled_canvas.Scroll(0,0)
			image_name=os.path.basename(path)
			if image_name not in self.information:
				self.information[image_name]={'polygons':[],'class_names':[]}
			self.current_polygon=[]
			self.canvas.Refresh()

			if self.AI_help:
				image=Image.open(path)
				image=np.array(image.convert('RGB'))
				self.sam2=self.sam2_model()
				self.sam2.set_image(image)


	def previous_image(self,event):

		if self.image_paths and self.current_image_id>0:
			self.current_image_id-=1
			self.load_current_image()
		self.canvas.SetFocus()


	def next_image(self,event):

		if self.image_paths and self.current_image_id<len(self.image_paths)-1:
			self.current_image_id+=1
			self.load_current_image()
		self.canvas.SetFocus()


	def delete_image(self,event):

		if self.image_paths:
			path=self.image_paths[self.current_image_id]
			self.image_paths.remove(path)
			image_name=os.path.basename(path)
			del self.information[image_name]
			self.load_current_image()
		self.canvas.SetFocus()


	def on_paint(self,event):

		if self.current_image is None:
			return

		dc=wx.PaintDC(self.canvas)
		dc.DrawBitmap(wx.Bitmap(self.current_image),0,0,True)
		image_name=os.path.basename(self.image_paths[self.current_image_id])
		polygons=self.information[image_name]['polygons']
		class_names=self.information[image_name]['class_names']

		if len(polygons)>0:
			for i,polygon in enumerate(polygons):
				color=self.color_map[class_names[i]]
				pen=wx.Pen(wx.Colour(*color),width=2)
				dc.SetPen(pen)
				dc.DrawLines(polygon)
				if self.start_modify:
					brush=wx.Brush(wx.Colour(*color))
					dc.SetBrush(brush)
					for x,y in polygon:
						dc.DrawCircle(x,y,4)
				if self.show_name:
					x_max=max(x for x,y in polygon)
					x_min=min(x for x,y in polygon)
					y_max=max(y for x,y in polygon)
					y_min=min(y for x,y in polygon)
					cx=int((x_max+x_min)/2)
					cy=int((y_max+y_min)/2)
					dc.SetTextForeground(wx.Colour(*color))
					dc.SetFont(wx.Font(wx.FontInfo(15).FaceName('Arial')))
					dc.DrawText(str(class_names[i]),cx,cy)

		if len(self.current_polygon)>0:
			current_polygon=[i for i in self.current_polygon]
			current_polygon.append(current_polygon[0])
			color=self.color_map[self.current_classname]
			brush=wx.Brush(wx.Colour(*color))
			dc.SetBrush(brush)
			for x,y in current_polygon:
				dc.DrawCircle(x,y,4)
			pen=wx.Pen(wx.Colour(*color),width=2)
			dc.SetPen(pen)
			dc.DrawLines(current_polygon)


	def on_left_click(self,event):

		x,y=event.GetX(),event.GetY()

		if self.start_modify:

			image_name=os.path.basename(self.image_paths[self.current_image_id])
			for i,polygon in enumerate(self.information[image_name]['polygons']):
				for j,(px,py) in enumerate(polygon):
					if abs(px-x)<5 and abs(py-y)<5:
						self.selected_point=(polygon,j,i)
						return

		else:

			if self.AI_help:
				self.foreground_points.append([x,y])
				points=self.foreground_points+self.background_points
				labels=[1 for i in range(len(self.foreground_points))]+[0 for i in range(len(self.background_points))]
				masks,scores,logits=self.sam2.predict(point_coords=np.array(points),point_labels=np.array(labels))
				mask=masks[np.argsort(scores)[::-1]][0]
				self.current_polygon=self.mask_to_polygon(mask)
			else:
				self.current_polygon.append((x,y))

		self.canvas.Refresh()


	def on_right_click(self,event):

		x,y=event.GetX(),event.GetY()

		if self.start_modify:

			return

		else:

			if len(self.current_polygon)>0:

				if self.AI_help:
					self.background_points.append([x,y])
					points=self.foreground_points+self.background_points
					labels=[1 for i in range(len(self.foreground_points))]+[0 for i in range(len(self.background_points))]
					masks,scores,logits=self.sam2.predict(point_coords=np.array(points),point_labels=np.array(labels))
					mask=masks[np.argsort(scores)[::-1]][0]
					self.current_polygon=self.mask_to_polygon(mask)
				else:
					self.current_polygon.pop()

			else:

				to_delete=[]
				image_name=os.path.basename(self.image_paths[self.current_image_id])
				polygons=self.information[image_name]['polygons']
				class_names=self.information[image_name]['class_names']
				if len(polygons)>0:
					for i,polygon in enumerate(polygons):
						x_max=max(x for x,y in polygon)
						x_min=min(x for x,y in polygon)
						y_max=max(y for x,y in polygon)
						y_min=min(y for x,y in polygon)
						if x_min<=x<=x_max and y_min<=y<=y_max:
							to_delete.append(i)
				if len(to_delete)>0:
					for i in sorted(to_delete,reverse=True):
						del self.information[image_name]['polygons'][i]
						del self.information[image_name]['class_names'][i]

		self.canvas.Refresh()


	def on_key_press(self,event):

		key_code=event.GetKeyCode()

		if event.GetKeyCode()==wx.WXK_RETURN:
			if len(self.current_polygon)>2:
				classnames=sorted(list(self.color_map.keys()))
				current_index=classnames.index(self.current_classname)
				dialog=wx.SingleChoiceDialog(self,message='Choose object class name',caption='Class Name',choices=classnames)
				dialog.SetSelection(current_index)
				if dialog.ShowModal()==wx.ID_OK:
					self.current_classname=dialog.GetStringSelection()
					if len(self.current_polygon)>0:
						self.current_polygon.append(self.current_polygon[0])
						image_name=os.path.basename(self.image_paths[self.current_image_id])
						self.information[image_name]['polygons'].append(self.current_polygon)
						self.information[image_name]['class_names'].append(self.current_classname)
				dialog.Destroy()
				self.current_polygon=[]
				self.foreground_points=[]
				self.background_points=[]
				self.canvas.Refresh()
		elif key_code==wx.WXK_LEFT:
			self.previous_image(None)
		elif key_code==wx.WXK_RIGHT:
			self.next_image(None)
		elif event.GetKeyCode()==wx.WXK_SHIFT:
			if self.start_modify:
				self.start_modify=False
			else:
				self.start_modify=True
			self.canvas.Refresh()
		elif event.GetKeyCode()==wx.WXK_SPACE:
			if self.show_name:
				self.show_name=False
			else:
				self.show_name=True
			self.canvas.Refresh()
		else:
			event.Skip()


	def on_left_move(self,event):

		if self.selected_point is not None and event.Dragging() and event.LeftIsDown():
			polygon,j,i=self.selected_point
			polygon[j]=event.GetPosition()
			image_name=os.path.basename(self.image_paths[self.current_image_id])
			self.information[image_name]['polygons'][i]=polygon
			self.canvas.Refresh()


	def on_left_up(self,event):

		self.selected_point=None


	def export_annotations(self,event):

		if not self.information:
			wx.MessageBox('No annotations to export.','Error',wx.ICON_ERROR)
			return

		self.generate_annotations(self.result_path,self.result_path,self.aug_methods)
		self.generate_annotations(os.path.dirname(self.image_paths[0]),self.result_path,[])

		wx.MessageBox('Annotations exported successfully.','Success',wx.ICON_INFORMATION)

		self.canvas.SetFocus()


	def generate_annotations(self,original_path,result_path,aug_methods):

		if not self.information:
			wx.MessageBox('No annotations to export.','Error',wx.ICON_ERROR)
			return

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

		for i,class_name in enumerate(sorted(list(self.color_map.keys()))):
			coco_format['categories'].append({
				'id':i+1,
				'name':class_name,
				'supercategory':'none'})

		image_id=0
		annotation_id=0
		parent_path=os.path.dirname(self.image_paths[0])

		for image_name in self.information:

			random.shuffle(methods)

			for m in methods:

				image=cv2.imread(os.path.join(parent_path,image_name))
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

				polygons=self.information[image_name]['polygons']

				if len(polygons)>0:

					for j,polygon in enumerate(self.information[image_name]['polygons']):

						category_id=sorted(list(self.color_map.keys())).index(self.information[image_name]['class_names'][j])+1
						segmentation=[np.array(polygon).flatten().tolist()]
						area=self.compute_area(polygon)
						bbox=self.compute_bbox(polygon)

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
										x,y=self.rotate_point(x,y,image_width/2,image_height/2,angle,image_width,image_height)
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
								rotated_box=np.array([self.rotate_point(px,py,image_width/2,image_height/2,angle,image_width,image_height) for px,py in box_points])
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


	def measure_annotations(self,event,threshold=None):

		if not self.information:
			wx.MessageBox('No annotations to measure.','Error',wx.ICON_ERROR)
			return

		data={}
		parameters=['center','area','height','width','perimeter','roundness','intensity']
		parent_path=os.path.dirname(self.image_paths[0])
		out_path=os.path.join(self.result_path,'Measurements')
		os.makedirs(out_path,exist_ok=True)

		for image_name in self.information:

			filename=os.path.splitext(image_name)[0]
			data[filename]={}
			for object_name in self.color_map:
				data[filename][object_name]={}
				for parameter in parameters:
					data[filename][object_name][parameter]=[]

			image=cv2.imread(os.path.join(parent_path,image_name))
			image_width=image.shape[1]
			image_height=image.shape[0]
			thickness=max(1,round(max(image_width,image_height)/960))
			to_annotate=image

			polygons=self.information[image_name]['polygons']

			if len(polygons)>0:

				for j,polygon in enumerate(self.information[image_name]['polygons']):

					mask=np.zeros((image_height,image_width),dtype=np.uint8)
					object_name=self.information[image_name]['class_names'][j]
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
							data[filename][object_name]['center'].append((int(cv2.moments(cnt)['m10']/cv2.moments(cnt)['m00']),int(cv2.moments(cnt)['m01']/cv2.moments(cnt)['m00'])))
							data[filename][object_name]['area'].append(area)
							data[filename][object_name]['height'].append(ht)
							data[filename][object_name]['width'].append(wd)
							data[filename][object_name]['perimeter'].append(perimeter)
							data[filename][object_name]['roundness'].append(roundness)
							data[filename][object_name]['intensity'].append(intensity)
							color=(self.color_map[object_name][2],self.color_map[object_name][1],self.color_map[object_name][0])
							if threshold is None:
								cv2.drawContours(to_annotate,[cnt],0,color,thickness)
							else:
								cv2.drawContours(to_annotate,sorted(cnts,key=cv2.contourArea,reverse=True)[:min(2,len(cnts))],-1,color,thickness)

			cv2.imwrite(os.path.join(out_path,filename+'_annotated.jpg'),to_annotate)

		with pd.ExcelWriter(os.path.join(out_path,'measurements.xlsx'),engine='openpyxl') as writer:

			for object_name in self.color_map:

				rows=[]
				columns=['filename','ID']+parameters

				for name,name_data in data.items():
					if object_name in name_data:
						values=zip(*[name_data[object_name][parameter] for parameter in parameters])
						for idx,value in enumerate(values):
							rows.append([name,idx+1]+list(value))

				df=pd.DataFrame(rows,columns=columns)
				df.to_excel(writer,sheet_name=object_name,float_format='%.2f',index=False)

		wx.MessageBox('Measurements exported successfully.','Success',wx.ICON_INFORMATION)

		self.canvas.SetFocus()


	def mask_to_polygon(self,mask):

		contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		if not contours:
			return []

		max_contour=max(contours,key=cv2.contourArea)
		epsilon=0.005*cv2.arcLength(max_contour,True)
		approx=cv2.approxPolyDP(max_contour,epsilon,True)

		return [tuple(point[0]) for point in approx]


	def compute_area(self,polygon):

		n=len(polygon)
		area=0

		for i in range(n):
			x1,y1=polygon[i]
			x2,y2=polygon[(i+1)%n]
			area+=x1*y2-x2*y1

		return abs(area)/2


	def compute_bbox(self,polygon):

		x_coords,y_coords=zip(*polygon)
		x_min=int(min(x_coords))
		y_min=int(min(y_coords))
		x_max=int(max(x_coords))
		y_max=int(max(y_coords))

		return [x_min,y_min,x_max-x_min,y_max-y_min]


	def rotate_point(self,x,y,cx,cy,angle,image_width,image_height):

		angle_rad=np.radians(-angle)
		cos_theta=np.cos(angle_rad)
		sin_theta=np.sin(angle_rad)
		x_shifted,y_shifted=x-cx,y-cy
		new_x=round(cos_theta*x_shifted-sin_theta*y_shifted+cx)
		new_y=round(sin_theta*x_shifted+cos_theta*y_shifted+cy)

		return max(0,min(image_width-1,new_x)),max(0,min(image_height-1,new_y))



class WindowLv2_AutoAnnotate(wx.Frame):

	def __init__(self,title):

		super(WindowLv2_AutoAnnotate,self).__init__(parent=None,title=title,size=(1000,260))
		self.path_to_images=None
		self.path_to_annotator=None
		self.object_kinds=None
		self.detection_threshold={}
		self.filters={}

		self.display_window()


	def display_window(self):

		panel=wx.Panel(self)
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_input=wx.BoxSizer(wx.HORIZONTAL)
		button_input=wx.Button(panel,label='Select the image(s)\nfor annotation',size=(300,40))
		button_input.Bind(wx.EVT_BUTTON,self.select_images)
		wx.Button.SetToolTip(button_input,'Select one or more images. Common image formats (jpg, png, tif) are supported. An annotation file will be generated in this folder')
		self.text_input=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_input.Add(button_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_input.Add(self.text_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_model=wx.BoxSizer(wx.HORIZONTAL)
		button_model=wx.Button(panel,label='Select a trained Annotator\nfor automatic annotation',size=(300,40))
		button_model.Bind(wx.EVT_BUTTON,self.select_model)
		wx.Button.SetToolTip(button_model,'A trained Annotator can annotate the objects of your interest in images.')
		self.text_model=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_model.Add(button_model,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_model.Add(self.text_model,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_model,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_filters=wx.BoxSizer(wx.HORIZONTAL)
		button_filters=wx.Button(panel,label='Specify the filters to\nexclude unwanted annotations',size=(300,40))
		button_filters.Bind(wx.EVT_BUTTON,self.specify_filters)
		wx.Button.SetToolTip(button_filters,'Select filters such as area, perimeter, roundness (1 is circle, higer value means less round), height, and width, and specify the minimum and maximum values of these filters.')
		self.text_filters=wx.StaticText(panel,label='Default: None',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_filters.Add(button_filters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_filters.Add(self.text_filters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_filters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_startannotation=wx.Button(panel,label='Start to annotate images',size=(300,40))
		button_startannotation.Bind(wx.EVT_BUTTON,self.start_annotation)
		wx.Button.SetToolTip(button_startannotation,'Automatically annotate objects in images.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_startannotation,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_images(self,event):

		wildcard='Image files(*.jpg;*.jpeg;*.png;*.tif;*.tiff)|*.jpg;*.jpeg;*.png;*.tif;*.tiff'
		dialog=wx.FileDialog(self,'Select images(s)','','',wildcard,style=wx.FD_MULTIPLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_images=dialog.GetPaths()
			self.path_to_images.sort()
			path=os.path.dirname(self.path_to_images[0])
			self.text_input.SetLabel('Select: '+str(len(self.path_to_images))+' images in'+str(path)+'.')
		dialog.Destroy()


	def select_model(self,event):

		annotator_path=os.path.join(the_absolute_current_path,'annotators')
		annotators=[i for i in os.listdir(annotator_path) if os.path.isdir(os.path.join(annotator_path,i))]
		if '__pycache__' in annotators:
			annotators.remove('__pycache__')
		if '__init__' in annotators:
			annotators.remove('__init__')
		if '__init__.py' in annotators:
			annotators.remove('__init__.py')
		annotators.sort()
		if 'Choose a new directory of the Annotator' not in annotators:
			annotators.append('Choose a new directory of the Annotator')

		dialog=wx.SingleChoiceDialog(self,message='Select an Annotator for automatic annotation.',caption='Select an Annotator',choices=annotators)
		if dialog.ShowModal()==wx.ID_OK:
			annotator=dialog.GetStringSelection()
			if annotator=='Choose a new directory of the Annotator':
				dialog1=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
				if dialog1.ShowModal()==wx.ID_OK:
					self.path_to_annotator=dialog1.GetPath()
				else:
					self.path_to_annotator=None
				dialog1.Destroy()
			else:
				self.path_to_annotator=os.path.join(annotator_path,annotator)
			with open(os.path.join(self.path_to_annotator,'model_parameters.txt')) as f:
				model_parameters=f.read()
			object_names=json.loads(model_parameters)['object_names']
			if len(object_names)>1:
				dialog1=wx.MultiChoiceDialog(self,message='Specify which obejct to annotate',
					caption='Object kind',choices=object_names)
				if dialog1.ShowModal()==wx.ID_OK:
					self.object_kinds=[object_names[i] for i in dialog1.GetSelections()]
				else:
					self.object_kinds=object_names
				dialog1.Destroy()
			else:
				self.object_kinds=object_names
			for object_name in self.object_kinds:
				dialog1=wx.NumberEntryDialog(self,'Detection threshold for '+str(object_name),'Enter an number between 0 and 100','Detection threshold for '+str(object_name),0,0,100)
				if dialog1.ShowModal()==wx.ID_OK:
					self.detection_threshold[object_name]=int(dialog1.GetValue())/100
				else:
					self.detection_threshold[object_name]=0
				dialog1.Destroy()
			self.text_model.SetLabel('Annotator: '+annotator+'; '+'The object kinds / detection threshold: '+str(self.detection_threshold)+'.')
		dialog.Destroy()

		if self.path_to_annotator is None:
			wx.MessageBox('No Annotator is selected.','No Annotator',wx.ICON_INFORMATION)
			self.text_model.SetLabel('No Annotator is selected.')


	def specify_filters(self,event):

		filters_choices=['area','perimeter','roundness','height','width']

		dialog=wx.MultiChoiceDialog(self,message='Select filters to exclude unwanted annotations',caption='Filters',choices=filters_choices)
		if dialog.ShowModal()==wx.ID_OK:
			selected_filters=[filters_choices[i] for i in dialog.GetSelections()]
		else:
			selected_filters=[]
		dialog.Destroy()

		for ft in selected_filters:
			dialog=wx.NumberEntryDialog(self,'The min value for '+str(ft),'The unit is pixel (except for roundness)','The min value for '+str(ft),0,0,100000000000000)
			values=[0,np.inf]
			if dialog.ShowModal()==wx.ID_OK:
				values[0]=int(dialog.GetValue())
			dialog.Destroy()
			dialog=wx.NumberEntryDialog(self,'The max value (enter 0 for infinity) for '+str(ft),'The unit is pixel (except for roundness)','The max value for '+str(ft),0,0,100000000000000)
			if dialog.ShowModal()==wx.ID_OK:
				value=int(dialog.GetValue())
				if value>0:
					values[1]=value
			dialog.Destroy()
			self.filters[ft]=values

		if len(self.filters)>0:
			self.text_filters.SetLabel('Filters: '+str(self.filters))
		else:
			self.text_filters.SetLabel('NO filters selected.')


	def start_annotation(self,event):

		if self.path_to_images is None or self.path_to_annotator is None:

			wx.MessageBox('No input images(s) / trained Annotator selected.','Error',wx.OK|wx.ICON_ERROR)

		else:
			
			AA=AutoAnnotation(self.path_to_images,self.path_to_annotator,self.object_kinds,detection_threshold=self.detection_threshold,filters=self.filters)
			AA.annotate_images()



def main_window():

	app=wx.App()
	InitialWindow(f'EZannot v{__version__}')
	print('The user interface initialized!')
	app.MainLoop()


if __name__=='__main__':

	main_window()

