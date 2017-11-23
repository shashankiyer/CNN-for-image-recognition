import sys
import os, errno
from PIL import Image, ImageDraw, ImageOps
import random
from math import sin, cos, fabs, pi, ceil, floor
import numpy as np



def initParam():
	global thicknessCircle, diameter, x1Circle, y1Circle, sizePlus, x1Plus, y1Plus, y2Plus, x2Plus, thicknessPlus, rotationPlus, arcWidth, arcHeight, thicknessWaves, leftCorner, rotationWave, sizeX, sizeY, side, thicknessSquare, rotationSquare, x1Square, y1Square, linelength, newx, newy, linewidth, rotationStar
	
	'''
	#parameters for Circle
	'''
	thicknessCircle = 3										#Thickess of the circle
	diameter = 10											#Diameter fo the circle
	x1Circle=int(random.randint(0,25-diameter))				#Top left corner of the circle's X value
	y1Circle=int(random.randint(0,25-diameter))				#Top left corner of the circle's Y value
	'''		
	'''		
			
	'''		
	#parameters for Plus Symbol		
	'''		
	sizePlus = 20											#width and height of Plus symbol
	x1Plus = random.randint(13, 38 - sizePlus)				#X coordinate of Top left corner of Plus symbol, before getting cropped
	y1Plus = random.randint(13, 38 - sizePlus)				#Y coordinate of Top left corner of Plus symbol, before getting cropped
	y2Plus = y1Plus + sizePlus								#Do not tweak
	x2Plus = x1Plus + sizePlus								#Do not tweak
			
	thicknessPlus = 3										#Thickness of Plus
			
	rotationPlus = 45										#Angle to rotate plus
	'''
	'''
	
	'''
	Parameters for wavy lines
	'''
	# Define the dimensions of the png image file before cropping it
	sizeX = 51  											#Size before cropping - Do not tweak
	sizeY = 51												#Size before cropping - Do not tweak
	
	#vary the size of the wavy lines (Total height of a curvy wave is arcHeight*3)
	arcWidth = 6    										#Width of arc
	arcHeight = 6   										#Height of Arc/3 (Value of 6 indicates height of 18)
	
	#vary the thickness of the symbol
	thicknessWaves = 1										#Thickness of Arc/3

	#vary the position of the symbol
	leftCorner = random.randint(13,sizeX-33)				#Left Corner X Coodinate

	#vary the rotation angle of the symbol
	rotationWave = random.randint(-90, 90)					#Degree of rotation for the wave
	'''
	'''
	
	''' 
	Parameters for Square
	'''
	side = 10												#Side of Square
	thicknessSquare = 3										#Thickness of square_draw
	rotationSquare = random.randint(0, 360)					#Angle of rotation of square
	
	x1Square = int(random.randint(ceil(13), ceil(25-side**(1/2)))) 		#This value should be between 13 and 25-sqrt(side)
	y1Square = int(random.randint(ceil(13), ceil(25-side**(1/2)))) 		#This value should be between 13 and 25-sqrt(side)
	'''
	'''

	'''
	Parameters for Star
	'''
	linelength = random.randint(5, 25)						#provide the chord length of the star
	newx =	random.randint(ceil(linelength/2), ceil(25-linelength/2))	#Offset x position for drawing the star
	newy =  random.randint(0, 25-linelength)				#Offset y position for drawing the star
	linewidth = 3											#Thickness of the lines
	rotationStar = random.randint(-90, 90)
	'''
	'''

#Tweakable parameters		
numberOfStrayMarks = 2
thicknessOfStrayMarks = 1
	
if(len(sys.argv)) != 3: 
	print ("Please supply the correct arguments")
	raise SystemExit(1)
	
folder_name = sys.argv[1]
num_examples = sys.argv[2]

try:
    os.makedirs(folder_name)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

#draw Stray Marks
def drawStrayMarks(numberOfStrayMarks, thicknessOfStrayMarks, draw):
	for x in range(0, numberOfStrayMarks):
			startPositionX = random.randint(0, 25-thicknessOfStrayMarks)
			startPositionY = random.randint(0, 25-thicknessOfStrayMarks)
			draw.ellipse([(startPositionX, startPositionY), (startPositionX+thicknessOfStrayMarks, startPositionY+thicknessOfStrayMarks)], None, 'black')


#draw Circles
for count in range(0, int(num_examples)):

	# size of image
	canvas = (25, 25)

	# scale ration
	scale = 1
	thumb = canvas[0]/scale, canvas[1]/scale
	
	initParam()
	
	y2Circle=diameter+y1Circle
	x2Circle=diameter+x1Circle

	
	circle_img = Image.new('L', ( 25 , 25 ), "white")
	circle_draw = ImageDraw.Draw(circle_img)
	
	for thick in range(0,thicknessCircle):
		xy = [(x1Circle,y1Circle),(x2Circle,y2Circle)]
		circle_draw.ellipse(xy, None, "black")
		x1Circle = x1Circle+1
		y1Circle = y1Circle+1
		x2Circle = x2Circle-1
		y2Circle = y2Circle-1
		
		
	
	drawStrayMarks(numberOfStrayMarks, thicknessOfStrayMarks, circle_draw)		#draw stray marks
	
	# make thumbnail
	circle_img.thumbnail(thumb)

	# save image
	circle_img.save(folder_name+'/' + str(count+1) + '_O' + '.png')
	
	

#P - for Plus symbol
for count in range(0, int(num_examples)):

	# size of image
	canvas = (25, 25)

	# scale ration
	scale = 1
	thumb = canvas[0]/scale, canvas[1]/scale
	
	initParam()
	
	cross_img = Image.new('L', ( 51 , 51 ), "white")
	cross_draw = ImageDraw.Draw(cross_img)
	xy = [ x1Plus , y1Plus + sizePlus/2 , x2Plus, y1Plus +sizePlus/2]
	cross_draw.line( xy, 0 , thicknessPlus)
	
	xy = [ x1Plus+sizePlus/2 , y1Plus, x1Plus+sizePlus/2 , y2Plus 	]
	cross_draw.line( xy, 0 , thicknessPlus)

	image = cross_img.rotate(rotationPlus)
	image = ImageOps.crop(image, 13)
	
	cross_draw = ImageDraw.Draw(image)
	drawStrayMarks(numberOfStrayMarks, thicknessOfStrayMarks, cross_draw)		#draw stray marks
	# make thumbnail
	image.thumbnail(thumb)

	# save image
	image.save(folder_name+'/' + str(count+1) + '_P' + '.png')
	

#W - for Waves
for count in range(0, int(num_examples)):

	# size of image
	canvas = (25, 25)

	# scale ration
	scale = 1
	thumb = canvas[0]/scale, canvas[1]/scale

	initParam()
	
	#create a new blank image
	image = Image.new('L', (sizeX, sizeY), 'white')
	draw = ImageDraw.Draw(image)

	for x in range(0, thicknessWaves):
		draw.arc((leftCorner, leftCorner+arcHeight, leftCorner+arcWidth, leftCorner+arcHeight*2), 270, 90, 'black')  # draw an arc in black
		draw.arc((leftCorner, leftCorner, leftCorner+arcWidth, leftCorner+arcHeight), 90, 270, 'black')  # draw an arc in black
		draw.arc((leftCorner, leftCorner+arcHeight*2, leftCorner+arcWidth, leftCorner+arcHeight*3), 90, 270, 'black')  # draw an arc in black

		draw.arc((leftCorner+arcWidth, leftCorner+arcHeight, leftCorner+arcWidth*2, leftCorner+arcHeight*2), 270, 90, 'black')  # draw an arc in black
		draw.arc((leftCorner+arcWidth, leftCorner, leftCorner+arcWidth*2, leftCorner+arcHeight), 90, 270, 'black')  # draw an arc in black
		draw.arc((leftCorner+arcWidth, leftCorner+arcHeight*2, leftCorner+arcWidth*2, leftCorner+arcHeight*3), 90, 270, 'black')  # draw an arc in black
	
		draw.arc((leftCorner+arcWidth*2, leftCorner+arcHeight, leftCorner+arcWidth*3, leftCorner+arcHeight*2), 270, 90, 'black')  # draw an arc in black
		draw.arc((leftCorner+arcWidth*2, leftCorner, leftCorner+arcWidth*3, leftCorner+arcHeight), 90, 270, 'black')  # draw an arc in black
		draw.arc((leftCorner+arcWidth*2, leftCorner+arcHeight*2, leftCorner+arcWidth*3, leftCorner+arcHeight*3), 90, 270, 'black')  # draw an arc in black
		leftCorner+=1

	image = image.rotate(rotationWave)
	image = ImageOps.crop(image, 13)
	
	draw = ImageDraw.Draw(image)
	
	"Till here"
	drawStrayMarks(numberOfStrayMarks, thicknessOfStrayMarks, draw)		#draw stray marks

	# make thumbnail
	image.thumbnail(thumb)

	# save image
	image.save(folder_name+'/' + str(count+1) + '_W' + '.png')
	
	

#Q - for Squares
for count in range(0, int(num_examples)):

	# size of image
	canvas = (25, 25)

	# scale ration
	scale = 1
	thumb = canvas[0]/scale, canvas[1]/scale
	
	initParam()
	"Definition here"	

	x2Square = side+x1Square
	y2Square = side+y1Square
	

	square_img = Image.new('L', (51, 51), "white")
	square_draw = ImageDraw.Draw(square_img)
	
	for thick in range(0, thicknessSquare):
		xy = [x1Square, y1Square, x2Square, y2Square]
		square_draw.rectangle(xy, "white", "black")
		x1Square = x1Square+1
		y1Square = y1Square+1
		x2Square = x2Square-1
		y2Square = y2Square-1
		
	square_img = square_img.rotate(rotationSquare)
	square_img = ImageOps.crop(square_img, 13)
	img = ImageDraw.Draw(square_img)
	
	"Till here"
	drawStrayMarks(numberOfStrayMarks, thicknessOfStrayMarks, img)		#draw stray marks
	# make thumbnail
	square_img.thumbnail(thumb)

	# save image
	square_img.save(folder_name+'/' + str(count+1) + '_Q' + '.png')
	

#S - for Stars`
for count in range(0, int(num_examples)):

	# size of image
	canvas = (25, 25)

	# scale ration
	scale = 1
	thumb = canvas[0]/scale, canvas[1]/scale
	
	"Definition here"
	im = Image.new('L', (51, 51), 'white')
	draw = ImageDraw.Draw(im)
	
	initParam()
	
	newx = newx+13
	newy = newy+13
	
	#Different parameters of the pentagram
	a = linelength/1.618
	b = linelength - a
	c = a - b
	
	#calculate points for the star and join them with each other
	pointA = [newx-b*(sin(18*pi/180)), newy+b*cos(18*pi/180)]
	draw.line((newx, newy, pointA[0], pointA[1]), fill=0, width = linewidth)
	
	pointB = [newx-a*cos(36*pi/180), newy+a*sin(36*pi/180)]
	draw.line((pointA[0], pointA[1], pointB[0], pointB[1]), fill=0, width = linewidth)
	
	pointC = [newx-a*sin(18*pi/180), newy+a*cos(18*pi/180)]
	draw.line((pointC[0], pointC[1], pointB[0], pointB[1]), fill=0, width = linewidth)
	
	pointD = [newx-linelength*sin(18*pi/180), newy+linelength*cos(18*pi/180)]
	draw.line((pointD[0], pointD[1], pointC[0], pointC[1]), fill=0, width = linewidth)
	
	pointE = [newx, newy+linelength-b*sin(36*pi/180)]
	draw.line((pointE[0], pointE[1], pointD[0], pointD[1]), fill=0, width = linewidth)
	
	pointF = [newx+b*cos(36*pi/180), newy+linelength*cos(18*pi/180)]
	draw.line((pointF[0], pointF[1], pointE[0], pointE[1]), fill=0, width = linewidth)
	
	pointG = [newx+a*sin(18*pi/180), newy+a*cos(18*pi/180)]
	draw.line((pointG[0], pointG[1], pointF[0], pointF[1]), fill=0, width = linewidth)
	
	pointH = [newx+a*cos(36*pi/180), newy+a*sin(36*pi/180)]
	draw.line((pointH[0], pointH[1], pointG[0], pointG[1]), fill=0, width = linewidth)
	
	pointI = [newx+b*(sin(18*pi/180)), newy+b*cos(18*pi/180)]
	draw.line((pointI[0], pointI[1], pointH[0], pointH[1]), fill=0, width = linewidth)
	
	draw.line((pointI[0], pointI[1], newx, newy), fill=0, width = linewidth)
	"Till here"
	im = im.rotate(rotationStar)
	im = ImageOps.crop(im, 13)
	
	draw = ImageDraw.Draw(im)
	drawStrayMarks(numberOfStrayMarks, thicknessOfStrayMarks, draw)		#draw stray marks
	
	# make thumbnail
	im.thumbnail(thumb)

	# save image
	im.save(folder_name+'/' + str(count+1) + '_S' + '.png')
