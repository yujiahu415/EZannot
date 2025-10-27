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
		button_outputfolder=wx.Button(panel,label='Select a folder to store the tiled\nimages and annotations',size=(300,40))
		button_outputfolder.Bind(wx.EVT_BUTTON,self.select_outpath)
		wx.Button.SetToolTip(button_outputfolder,'Copies of tiled images and the corresponding annotation file(s) will be stored in this folder.')
		self.text_outputfolder=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_outputfolder.Add(button_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_outputfolder.Add(self.text_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_outputfolder,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_parameters=wx.BoxSizer(wx.HORIZONTAL)
		button_parameters=wx.Button(panel,label='Specify the parameters for\ntiling the imagess',size=(300,40))
		button_parameters.Bind(wx.EVT_BUTTON,self.specify_parameters)
		wx.Button.SetToolTip(button_parameters,'Specify the tiling parameters such as tile size, overlapping ratio, and whether the background is black.')
		self.text_parameters=wx.StaticText(panel,label='Default: tile size=(640,640), overlapping ratio=0.2, black background',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_parameters.Add(button_parameters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_parameters.Add(self.text_parameters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_parameters,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_starttile=wx.Button(panel,label='Start to tile images',size=(300,40))
		button_starttile.Bind(wx.EVT_BUTTON,self.start_tiling)
		wx.Button.SetToolTip(button_starttile,'Make the annotated images into smaller tiles.')
		boxsizer.Add(0,5,0)
		boxsizer.Add(button_starttile,0,wx.RIGHT|wx.ALIGN_RIGHT,90)
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


	def specify_parameters(self,event):

		outtext='Tile size: '

		dialog=wx.NumberEntryDialog(self,'Input the size\nof each tile','A number divisible by 32 (min 320 max 6400):','Tile size',640,1,6400)
		if dialog.ShowModal()==wx.ID_OK:
			self.tile_size=(int(dialog.GetValue()),int(dialog.GetValue()))
		else:
			self.tile_size=(640,640)
		dialog.Destroy()
		outtext=outtext+str(self.tile_size)+', overlapping ratio: '

		dialog=wx.NumberEntryDialog(self,'Input the overlapping ratio\nbetween adjacent tiles','A number between 1 and 100:','Overlapping ratio',20,1,100)
		if dialog.ShowModal()==wx.ID_OK:
			self.overlap_ratio=int(dialog.GetValue())/100
		else:
			self.overlap_ratio=0.2
		dialog.Destroy()
		outtext=outtext+str(self.overlap_ratio)+', '

		dialog=wx.MessageDialog(self,'Is the background in the images black or darker than foreground?','Darker background?',wx.YES_NO|wx.ICON_QUESTION)
		if dialog.ShowModal()==wx.ID_YES:
			self.black_background=True
			outtext=outtext+'black background'+'.'
		else:
			self.black_background=False
			outtext=outtext+'white background'+'.'
		dialog.Destroy()

		self.text_parameters.SetLabel(outtext)


	def start_tiling(self,event):

		if self.path_to_images is None or self.out_path is None:

			wx.MessageBox('No input / output folder.','Error',wx.OK|wx.ICON_ERROR)

		else:
			
			tile_annotation(self.path_to_images,self.out_path,tile_size=self.tile_size,overlap_ratio=self.overlap_ratio,black_background=self.black_background)


