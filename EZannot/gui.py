import os
import cv2
import wx
import wx.lib.agw.hyperlink as hl
import json
import random
import itertools
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import rotate
from PIL import Image
from screeninfo import get_monitors
from EZannot.sam2.build_sam import build_sam2
from EZannot.sam2.sam2_image_predictor import SAM2ImagePredictor
from EZannot import __version__



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

		super(InitialWindow,self).__init__(parent=None,title=title,size=(750,440))
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
		button_startannotate=wx.Button(panel,label='Start Annotation',size=(200,40))
		button_startannotate.Bind(wx.EVT_BUTTON,self.start_annotate)
		wx.Button.SetToolTip(button_startannotate,'Start to annotate images with ease.')
		module_modules.Add(button_startannotate,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_modules,0,wx.ALIGN_CENTER,50)
		boxsizer.Add(0,50,0)

		panel.SetSizer(boxsizer)

		self.Centre()
		self.Show(True)


	def start_annotate(self,event):

		WindowLv1_SetAnnotation('Start Annotation')



class WindowLv1_SetAnnotation(wx.Frame):

	def __init__(self,title):

		super(WindowLv1_SetAnnotation,self).__init__(parent=None,title=title,size=(1000,370))
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
		wx.Button.SetToolTip(button_input,'Select one or more images. Common image formats (jpg, png, tif) are supported.')
		self.text_input=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_input.Add(button_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_input.Add(self.text_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,10,0)
		boxsizer.Add(module_input,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		module_outputfolder=wx.BoxSizer(wx.HORIZONTAL)
		button_outputfolder=wx.Button(panel,label='Select a folder to store\nthe annotated images',size=(300,40))
		button_outputfolder.Bind(wx.EVT_BUTTON,self.select_outpath)
		wx.Button.SetToolTip(button_outputfolder,'Images and the annotation file will be stored in this folder.')
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
		wx.Button.SetToolTip(button_augmentation,'Use augmentation for the annotated images can greatly enhance the training efficiency.')
		self.text_augmentation=wx.StaticText(panel,label='None.',style=wx.ALIGN_LEFT|wx.ST_ELLIPSIZE_END)
		module_augmentation.Add(button_augmentation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		module_augmentation.Add(self.text_augmentation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(module_augmentation,0,wx.LEFT|wx.RIGHT|wx.EXPAND,10)
		boxsizer.Add(0,5,0)

		button_startannotation=wx.Button(panel,label='Start to annotate images',size=(300,40))
		button_startannotation.Bind(wx.EVT_BUTTON,self.start_annotation)
		wx.Button.SetToolTip(button_startannotation,'Annotate objects in images.')
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

		dialog=wx.TextEntryDialog(self,'Enter the names of objects to annotate\n(use "," to separate each name)','Object class names')
		color_map={}
		if dialog.ShowModal()==wx.ID_OK:
			entry=dialog.GetValue()
			try:
				for i in entry.split(','):
					color_map[i]='#%02x%02x%02x'%(random.randint(0,255),random.randint(0,255),random.randint(0,255))
			except:
				color_map={}
				wx.MessageBox('Please enter the object class names in\ncorrect format! For example: apple,orange,pear','Error',wx.OK|wx.ICON_ERROR)
		dialog.Destroy()

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
			WindowLv2_AnnotateImages(None,'Annotate Images',self.path_to_images,self.result_path,self.color_map,self.aug_methods,model_cp=self.model_cp,model_cfg=self.model_cfg)



class WindowLv2_AnnotateImages(wx.Frame):

	def __init__(self,parent,title,path_to_images,result_path,color_map,aug_methods,model_cp=None,model_cfg=None):

		monitor=get_monitors()[0]
		monitor_w,monitor_h=monitor.width,monitor.height

		super().__init__(parent,title=title,pos=(75,0),size=(get_monitors()[0].width-150,get_monitors()[0].height-150))

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
		self.AI_help=False

		annotation_file=None
		for i in os.listdir(os.path.dirname(self.image_paths[0])):
			if i.endswith('.json'):
				annotation_file=os.path.join(os.path.dirname(self.image_paths[0]),i)
		if annotation_file and os.path.exists(annotation_file):
			colors=[self.color_map[i] for i in self.color_map]
			classnames=[]
			annotation=json.load(open(annotation_file))
			for i in annotation['categories']:
				if i['id']>0:
					classnames.append(i['name'])
			len_diff=len(classnames)-len(self.color_map)
			self.current_classname=classnames[0]
			if len_diff>0:
				for i in range(len_diff):
					colors.append((random.randint(0,255),random.randint(0,255),random.randint(0,255)))
			self.color_map={}
			for i,classname in enumerate(classnames):
				self.color_map[classname]=colors[i]
			for i in annotation['images']:
				self.information[i['file_name']]={'polygons':[],'class_names':[]}
			for i in annotation['annotations']:
				image_name=annotation['images'][int(i['image_id'])]['file_name']
				classname=annotation['categories'][int(i['category_id'])-1]['name']
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

		self.export_button=wx.Button(panel,label='Export Annotations',size=(150,30))
		self.export_button.Bind(wx.EVT_BUTTON,self.export_annotations)
		hbox.Add(self.export_button,flag=wx.ALL,border=2)

		self.measure_button=wx.Button(panel,label='Measure Annotations',size=(150,30))
		self.measure_button.Bind(wx.EVT_BUTTON,self.measure_annotations)
		hbox.Add(self.measure_button,flag=wx.ALL,border=2)
		vbox.Add(hbox,flag=wx.ALIGN_CENTER|wx.TOP,border=5)

		self.scrolled_canvas=wx.ScrolledWindow(panel,style=wx.VSCROLL|wx.HSCROLL)
		self.scrolled_canvas.SetScrollRate(10,10)
		self.canvas=wx.Panel(self.scrolled_canvas,pos=(75,0),size=(get_monitors()[0].width-150,get_monitors()[0].height-150))
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

		remove=[]

		all_methods=['','_rot1','_rot2','_rot3','_rot4','_rot5','_rot6','_blur']
		options=['_rot7','_flph','_flpv','_brih','_bril','_exph','_expl']
		for r in range(1,len(options)+1):
			all_methods.extend([''.join(c) for c in itertools.combinations(options,r)])

		for i in all_methods:
			if 'random rotation' not in self.aug_methods:
				if 'rot' in i:
					remove.append(i)
			if 'horizontal flipping' not in self.aug_methods:
				if 'flph' in i:
					remove.append(i)
			if 'vertical flipping' not in self.aug_methods:
				if 'flpv' in i:
					remove.append(i)
			if 'random brightening' not in self.aug_methods:
				if 'brih' in i:
					remove.append(i)
				if 'exph' in i:
					remove.append(i)
			if 'random dimming' not in self.aug_methods:
				if 'bril' in i:
					remove.append(i)
				if 'expl' in i:
					remove.append(i)
			if 'random blurring' not in self.aug_methods:
				if 'blur' in i:
					remove.append(i)

		methods=list(set(all_methods)-set(remove))

		coco_format={'categories':[],'images':[],'annotations':[]}

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
				cv2.imwrite(os.path.join(self.result_path,new_name),image)

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

		with open(os.path.join(self.result_path,'annotations.json'),'w') as json_file:
			json.dump(coco_format,json_file)
		wx.MessageBox('Annotations exported successfully.','Success',wx.ICON_INFORMATION)

		self.canvas.SetFocus()


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
			for cell_name in self.color_map:
				data[filename][cell_name]={}
				for parameter in parameters:
					data[filename][cell_name][parameter]=[]

			image=cv2.imread(os.path.join(parent_path,image_name))
			image_width=image.shape[1]
			image_height=image.shape[0]
			thickness=max(1,round(max(image_width,image_height)/960))
			to_annotate=image

			polygons=self.information[image_name]['polygons']

			if len(polygons)>0:

				for j,polygon in enumerate(self.information[image_name]['polygons']):

					mask=np.zeros((image_height,image_width),dtype=np.uint8)
					cell_name=self.information[image_name]['class_names'][j]
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
						roundness=(4*np.pi*area)/(perimeter*perimeter)
						(_,_),(wd,ht),_=cv2.minAreaRect(cnt)
						intensity=(np.sum(image*cv2.cvtColor(mask*255,cv2.COLOR_GRAY2BGR))/3)/max(area,1)
						if area>0:
							data[filename][cell_name]['center'].append((int(cv2.moments(cnt)['m10']/cv2.moments(cnt)['m00']),int(cv2.moments(cnt)['m01']/cv2.moments(cnt)['m00'])))
							data[filename][cell_name]['area'].append(area)
							data[filename][cell_name]['height'].append(ht)
							data[filename][cell_name]['width'].append(wd)
							data[filename][cell_name]['perimeter'].append(perimeter)
							data[filename][cell_name]['roundness'].append(roundness)
							data[filename][cell_name]['intensity'].append(intensity)
							color=(self.color_map[cell_name][2],self.color_map[cell_name][1],self.color_map[cell_name][0])
							if threshold is None:
								cv2.drawContours(to_annotate,[cnt],0,color,thickness)
							else:
								cv2.drawContours(to_annotate,sorted(cnts,key=cv2.contourArea,reverse=True)[:min(2,len(cnts))],-1,color,thickness)

			cv2.imwrite(os.path.join(out_path,filename+'_annotated.jpg'),to_annotate)

		with pd.ExcelWriter(os.path.join(out_path,'measurements.xlsx'),engine='openpyxl') as writer:

			for cell_name in self.color_map:

				rows=[]
				columns=['filename','ID']+parameters

				for name,name_data in data.items():
					if cell_name in name_data:
						values=zip(*[name_data[cell_name][parameter] for parameter in parameters])
						for idx,value in enumerate(values):
							rows.append([name,idx+1]+list(value))

				df=pd.DataFrame(rows,columns=columns)
				df.to_excel(writer,sheet_name=cell_name,float_format='%.2f',index=False)

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



def main_window():

	app=wx.App()
	InitialWindow(f'EZannot v{__version__}')
	print('The user interface initialized!')
	app.MainLoop()


if __name__=='__main__':

	main_window()

