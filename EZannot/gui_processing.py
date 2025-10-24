import os
import cv2
import wx
import json
import random
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
from .gui_training import PanelLv1_TrainingModule
from .gui_annotating import PanelLv1_AnnotationModule
from .annotator import Annotator,AutoAnnotation
from .tools import read_annotation,measure_annotation,tile_annotation



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



class PanelLv1_ProcessingModule(wx.Panel):

	def __init__(self,parent):

		super().__init__(parent)
		self.notebook=parent
		self.dispaly_window()


	def dispaly_window(self):

		panel=self
		boxsizer=wx.BoxSizer(wx.VERTICAL)
		boxsizer.Add(0,60,0)

		button_measureannotations=wx.Button(panel,label='Measure Annotations',size=(300,40))
		button_measureannotations.Bind(wx.EVT_BUTTON,self.measure_annotations)
		wx.Button.SetToolTip(button_measureannotations,'Automatically provide quantitative measurements (such as center,area,height,width,perimeter,roundness,intensity) for each annotated object.')
		boxsizer.Add(button_measureannotations,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,5,0)

		button_tileannotations=wx.Button(panel,label='Tile Annotations',size=(300,40))
		button_tileannotations.Bind(wx.EVT_BUTTON,self.tile_annotations)
		wx.Button.SetToolTip(button_tileannotations,'Divide large annotated images into smaller tiles and preserve the annotations.')
		boxsizer.Add(button_tileannotations,0,wx.ALIGN_CENTER,10)
		boxsizer.Add(0,50,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def measure_annotations(self,event):

		panel=PanelLv2_MeasureAnnotations(self.notebook)
		title='Measure Annotations'
		self.notebook.AddPage(panel,title,select=True)


	def tile_annotations(self,event):

		panel=PanelLv2_TileAnnotations(self.notebook)
		title='Tile Annotations'
		self.notebook.AddPage(panel,title,select=True)



class PanelLv2_MeasureAnnotations(wx.Panel):

	def __init__(self,parent):

		super().__init__(parent)
		self.notebook=parent
		self.path_to_images=None
		self.result_path=None
		self.color_map={}
		self.show_ids=False

		self.display_window()


	def display_window(self):

		panel=self
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_input=wx.BoxSizer(wx.HORIZONTAL)
		button_input=wx.Button(panel,label='Select the folder that stores\nannotated images for measurements',size=(300,40))
		button_input.Bind(wx.EVT_BUTTON,self.select_inpath)
		wx.Button.SetToolTip(button_input,'Select the folder that stores all the annotated images. You also need to put the annotation file(s) in the same folder, and EZannot will decode the annotations in the annotation file(s) automatically.')
		self.text_input=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_input.Add(button_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_input.Add(self.text_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_outputfolder=wx.BoxSizer(wx.HORIZONTAL)
		button_outputfolder=wx.Button(panel,label='Select a folder to store\nthe measurements',size=(300,40))
		button_outputfolder.Bind(wx.EVT_BUTTON,self.select_outpath)
		wx.Button.SetToolTip(button_outputfolder,'Copies of annotated images and the quantitative measurements will be stored in this folder.')
		self.text_outputfolder=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_outputfolder.Add(button_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_outputfolder.Add(self.text_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_classes=wx.BoxSizer(wx.HORIZONTAL)
		button_classes=wx.Button(panel,label='Specify the colors for\nthe annotated objects',size=(300,40))
		button_classes.Bind(wx.EVT_BUTTON,self.specify_colors)
		wx.Button.SetToolTip(button_classes,'Specify the colors that represent the annotated objects and EZannot will output the annotated images.')
		self.text_classes=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_classes.Add(button_classes,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_classes.Add(self.text_classes,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_classes,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_measureannotation=wx.Button(panel,label='Measure the annotated',size=(300,40))
		button_measureannotation.Bind(wx.EVT_BUTTON,self.measure_annotations)
		wx.Button.SetToolTip(button_measureannotation,'Calculate diverse quantitative measurements for each annotated object.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_measureannotation,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_inpath(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_images=dialog.GetPath()
			self.text_input.SetLabel('The folder that stores all the annotated images: '+self.path_to_images+'.')
		dialog.Destroy()


	def select_outpath(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.result_path=dialog.GetPath()
			self.text_outputfolder.SetLabel('The annotated images will be in: '+self.result_path+'.')
		dialog.Destroy()


	def specify_colors(self,event):

		if self.path_to_images is None:

			wx.MessageBox('No input images(s).','Error',wx.OK|wx.ICON_ERROR)

		else:

			annotation_files=[]
			color_map={}
			self.color_map={}
			classnames=[]

			for i in os.listdir(self.path_to_images):
				if i.endswith('.json'):
					annotation_files.append(os.path.join(self.path_to_images,i))

			if len(annotation_files)>0:
				for annotation_file in annotation_files:
					if os.path.exists(annotation_file):
						annotation=json.load(open(annotation_file))
						for i in annotation['categories']:
							if i['id']>0:
								classname=i['name']
								if classname not in classnames:
									classnames.append(classname)

			for i in classnames:
				color_map[i]='#%02x%02x%02x'%(random.randint(0,255),random.randint(0,255),random.randint(0,255))

			if len(color_map)>0:
				for classname in color_map:
					dialog=ColorPicker(self,f'{classname}',[classname,color_map[classname]])
					if dialog.ShowModal()==wx.ID_OK:
						(r,b,g,_)=dialog.color_picker.GetColour()
						self.color_map[classname]=(r,b,g)
					dialog.Destroy()
				self.text_classes.SetLabel('Classname:color: '+str(self.color_map)+'.')
			else:
				self.text_classes.SetLabel('None.')

			dialog=wx.MessageDialog(self,'Show the IDs for\nall the annotated objects?','Show IDs?',wx.YES_NO|wx.ICON_QUESTION)
			if dialog.ShowModal()==wx.ID_YES:
				self.show_ids=True
			else:
				self.show_ids=False
			dialog.Destroy()


	def measure_annotations(self,event):

		if self.path_to_images is None or self.result_path is None:

			wx.MessageBox('No input / output folder.','Error',wx.OK|wx.ICON_ERROR)

		else:

			information=read_annotation(self.path_to_images)

			measure_annotation(self.path_to_images,self.result_path,information,self.color_map,show_ids=self.show_ids,threshold=None)



class PanelLv2_TileAnnotations(wx.Panel):

	def __init__(self,parent):

		super().__init__(parent)
		self.notebook=parent
		self.path_to_images=None
		self.out_path=None
		self.tile_size=(640,640)
		self.overlap_ratio=0.2
		self.black_background=False

		self.display_window()


	def display_window(self):

		panel=self
		boxsizer=wx.BoxSizer(wx.VERTICAL)

		module_input=wx.BoxSizer(wx.HORIZONTAL)
		button_input=wx.Button(panel,label='Select the folder that stores\nannotated images for tiling',size=(300,40))
		button_input.Bind(wx.EVT_BUTTON,self.select_inpath)
		wx.Button.SetToolTip(button_input,'Select the folder that stores all the annotated images. You also need to put the annotation file(s) in the same folder, and EZannot will decode the annotations in the annotation file(s) automatically.')
		self.text_input=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_input.Add(button_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_input.Add(self.text_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_outputfolder=wx.BoxSizer(wx.HORIZONTAL)
		button_outputfolder=wx.Button(panel,label='Select a folder to store\nthe tiled annotations',size=(300,40))
		button_outputfolder.Bind(wx.EVT_BUTTON,self.select_outpath)
		wx.Button.SetToolTip(button_outputfolder,'Copies of tiled images and the corresponding annotation file(s) will be stored in this folder.')
		self.text_outputfolder=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_outputfolder.Add(button_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_outputfolder.Add(self.text_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_filters=wx.BoxSizer(wx.HORIZONTAL)
		button_filters=wx.Button(panel,label='Specify the parameters for\ntiling the imagess',size=(300,40))
		button_filters.Bind(wx.EVT_BUTTON,self.specify_parameters)
		wx.Button.SetToolTip(button_filters,'Specify the tiling parameters such as tile size, overlapping ratio, and whether the background is black.')
		self.text_filters=wx.StaticText(panel,label='Default: tile size=(640,640), overlapping ratio=0.2, black background',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_filters.Add(button_filters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_filters.Add(self.text_filters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_filters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_startannotation=wx.Button(panel,label='Start to tile images',size=(300,40))
		button_startannotation.Bind(wx.EVT_BUTTON,self.start_tiling)
		wx.Button.SetToolTip(button_startannotation,'Make the annotated images into smaller tiles.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_startannotation,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
		boxsizer.Add(0,10,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def select_inpath(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.path_to_images=dialog.GetPath()
			self.text_input.SetLabel('The folder that stores all the annotated images: '+self.path_to_images+'.')
		dialog.Destroy()


	def select_outpath(self,event):

		dialog=wx.DirDialog(self,'Select a directory','',style=wx.DD_DEFAULT_STYLE)
		if dialog.ShowModal()==wx.ID_OK:
			self.out_path=dialog.GetPath()
			self.text_outputfolder.SetLabel('The tiled annotated images will be in: '+self.out_path+'.')
		dialog.Destroy()


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



class MainFrame(wx.Frame):

	def __init__(self):
		super().__init__(None,title=f'EZannot v{__version__}')
		self.SetSize((1000,500))

		self.aui_manager=wx.aui.AuiManager()
		self.aui_manager.SetManagedWindow(self)

		self.notebook=wx.aui.AuiNotebook(self)
		self.aui_manager.AddPane(self.notebook,wx.aui.AuiPaneInfo().CenterPane())

		panel=InitialPanel(self.notebook)
		title='Welcome'
		self.notebook.AddPage(panel,title,select=True)

		sizer=wx.BoxSizer(wx.VERTICAL)
		sizer.Add(self.notebook,1,wx.EXPAND)
		self.SetSizer(sizer)

		self.aui_manager.Update()
		self.Centre()
		self.Show()



def main_window():

	app=wx.App()
	MainFrame()
	print('The user interface initialized!')
	app.MainLoop()


if __name__=='__main__':

	main_window()

