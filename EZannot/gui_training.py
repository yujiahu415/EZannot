import os
import cv2
import wx
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from .annotator import Annotator



the_absolute_current_path=str(Path(__file__).resolve().parent)



class PanelLv1_TrainingModule(wx.Panel):

	def __init__(self,parent):

		super().__init__(parent)
		self.notebook=parent
		self.dispaly_window()


	def dispaly_window(self):

		panel=self
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

		panel=PanelLv2_TrainAnnotators(self.notebook)
		title='Train Annotators'
		self.notebook.AddPage(panel,title,select=True)


	def test_annotators(self,event):

		panel=PanelLv2_TestAnnotators(self.notebook)
		title='Test Annotators'
		self.notebook.AddPage(panel,title,select=True)



class PanelLv2_TrainAnnotators(wx.Panel):

	def __init__(self,parent):

		super().__init__(parent)
		self.notebook=parent
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

		panel=self
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



class PanelLv2_TestAnnotators(wx.Panel):

	def __init__(self,parent):

		super().__init__(parent)
		self.notebook=parent
		self.path_to_testingimages=None
		self.path_to_annotation=None
		self.annotator_path=os.path.join(the_absolute_current_path,'annotators')
		self.path_to_annotator=None
		self.output_path=None

		self.dispaly_window()


	def dispaly_window(self):

		panel=self
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


