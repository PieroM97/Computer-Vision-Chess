"""
Alcune delle funzioni che utilizzerò all'interno del programma
sono presenti all'interno delle librerie da me definite :

- MyChessFunction , Contente le funzioni che fungono da interfaccia
tra OpenCV e il sistema di scacchi
-Calibration , per la calibrazione della camera e la creazione di set
per la medesima funzione
- TeachableMachine , contente la parte relativa al machine learning
"""
# noinspection PyUnresolvedReferences
from MyChessFunction import *
# noinspection PyUnresolvedReferences
from Calibration import *

from teachableMachine import *
from helper import *

"""
Librerie esterne in uso all'interno del programma : 
-chess e chess.engine per la parte legata al motore di scacchi
-stockfish per le partite contro la CPU  
"""
import chess
import chess.engine
import chess.svg
from stockfish import Stockfish

"""Tutte le librerie cui sotto sono state utilizzate per la
definizione dell'interfaccia e delle sue parti : 
Ho preferito utilizzare PyQt5 per la compatibilità rispetto ai file
SVG , formato nel quale la scacchiera viene codificata"""
from PyQt5.QtSvg import QSvgWidget
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton, QMessageBox, QToolTip, QMenuBar, \
	QMenu, QAction, QFrame, QProgressBar, QSlider, QComboBox, QRadioButton, QGridLayout, QGroupBox, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QTimer, QPropertyAnimation, QRect

"""Altre librerie utilizzate"""
import time
import sys


"""Gestione ed eventuale creazione delle cartelle per il dataset"""
from pathlib import Path
Path("new_chessboard/Empty").mkdir(parents=True, exist_ok=True)
Path("new_chessboard/Black").mkdir(parents=True, exist_ok=True)
Path("new_chessboard/White").mkdir(parents=True, exist_ok=True)


"""Classe per la gestione della visione artificiale e per
la elaborazione delle immagini """
# noinspection PyUnresolvedReferences
import cv2

"""Classe per la gestione delle matrici presenti all'interno 
del programma e gestione numerica di una serie di funzioni """
import numpy as np

"""Utilizzato per la gestione della Splash Screen"""
from datetime import time

"""Importazione pacchetti utili :
-Tensorflow
-Keras
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""Per gestire le immagini presenti nelle cartelle"""
import pathlib

"""Variabili utili per il corretto funzionamento del programma :
per lo più flag , ma anche variabili globali che ho usato all'interno 
delle funzioni di gioco """

# Flag che indica se la scacchiera è stata trovata
chessboard_found = False
# Flag che indica se le case della scacchiera sono state trovate
boxes_found = False
# Variabile contente le coordinate dei punti che costiuiscono le case
chessboard_corners = None
# Variabile contente i quattro vertici della scacchiera
box_corners = None
# Immagine della scacchiera ( da webcam )
img_chessboard = None
# Variabile contente tutte le coordinate della scacchiera ( case comprese )
coordinates = None
# Flag per la gestione dei turni
isWhiteTurn = True
isBlackTurn = False
# Matrice per lo stato S0 per quanto riguarda entrambi i lati di gioco
oldblack = setBlack()
oldwhite = setWhite()
# Oggetto di tipo scacchiera , contenente lo stato di gioco
chessboard = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")
# Motore Stockfish , per la scelta delle mosse migliori e per la gestione delle partite vs CPU
stockfish = Stockfish("stockfish-10-win\Windows\stockfish_10_x64")
# Flag da utilizzare nella procedura per la creazione della scacchiera
step = 0
#Flag che indica se il programma sta caricando
isLoading = True
#Per la creazione di un modello , per la barra di caricamento
refApp = None
#Variabile che indica il numero di epoche per la creazione del modello
epochs = 20
#Flag che indica se si sta utilizzando la scacchiera di default o meno
chessboard_type = 0
#Flag che indica se è stato creato un modello personale
isAvailable = False

"""Thread per la creazione del modello"""
class TaskThread(QThread):
	taskFinished = pyqtSignal()

	# Costruttore della classe
	def __init__(self):
		super().__init__()
		self._run_flag = True

	def run(self):
		while self._run_flag:
			# TEST DIFFERENT MACHINE LEARNING MODEL

			data_dir = pathlib.Path("new_chessboard")

			image_count = len(list(data_dir.glob('*/*.jpg')))

			print("Il dataset è costituito da " + str(image_count) + " immagini.")

			"""Parametri per keras """

			batch_size = 16
			width = 224
			height = 224

			"""Training"""
			train_ds = tf.keras.preprocessing.image_dataset_from_directory(
				data_dir,
				validation_split=0.2,
				subset="training",
				seed=123,
				image_size=(height, width),
				batch_size=batch_size)

			"""Validation"""
			val_ds = tf.keras.preprocessing.image_dataset_from_directory(
				data_dir,
				validation_split=0.2,
				subset="validation",
				seed=123,
				image_size=(height, width),
				batch_size=batch_size)




			####################################################################

			"""Configurazione dei dati per migliorarne le prestazioni """

			train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
			val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

			model = tf.keras.applications.MobileNet(
				input_shape=None, alpha=1.0, depth_multiplier=1, dropout=0.001,
				include_top=True, weights="imagenet", input_tensor=None, pooling=None,
				classes=1000, classifier_activation='softmax'
			)

			model.compile(
				optimizer='adam',
				loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
				metrics=['accuracy'])

			global epochs

			history = model.fit(
				train_ds,
				validation_data=val_ds,
				epochs=epochs,
				verbose=0,
				callbacks=[CustomCallback()]
			)

			acc = history.history['accuracy']
			val_acc = history.history['val_accuracy']

			print(acc)
			print(val_acc)


			model.save("my_keras.h5")

			self._run_flag = False
			self.taskFinished.emit()

	# Quando il programma viene terminato
	def stop(self):
		self._run_flag = False
		self.wait()

"""Callback per ottenere l'epoca all'interno 
del processo che porta alla creazione del modello"""
class CustomCallback(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		global refApp
		global epochs
		refApp.progressBar.setValue(100/epochs*(epoch+1))


"""Classe che estende un Thread per l'utilizzo della schermata
della webcam , che riprende la scacchiera , all'interno 
dell'interfaccia . Oltre al costruttore , sono presenti la funzione 
di stop per chiudere la finestra e quella di run , che elabora 
l'immagine e ottiene le informazioni utili relative al posizionamento 
della scacchiera e delle case """


class VideoThread(QThread):
	change_pixmap_signal = pyqtSignal(np.ndarray)

	# Costruttore della classe
	def __init__(self):
		super().__init__()
		self._run_flag = True

	# Corpo del thread
	def run(self):

		set_model(0)

		# Gestione webcam
		webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

		# Calibrazione webcam
		ret, mtx, dist, rvecs, tvecs, h, w = camera_calibration()
		newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

		# Ciclo per la cattura dei frame
		while self._run_flag:
			success, img = webcam.read()

			if success:

				# Rimuovere distorsione
				img = undistort(img, mtx, dist, newcameramtx, roi)

				# Pre-Elaborazione dell'immagine
				img_ = pre_processing(img)

				# Definizione delle variabili globali da utilizzare nel corpo del metodo
				global chessboard_found
				global chessboard_corners

				# Nel caso la scacchiera è stata già trovata , la ricerca viene conclusa
				if (chessboard_found == False):
					# Ricerca della scacchiera
					chessboard_found, chessboard_corners = cv2.findChessboardCorners(img_, (7, 7), None)

				# Se la scacchiera è stata trovata
				if chessboard_found:

					#draw_chessboard_sides(img, chessboard_found, chessboard_corners)
					# Variabile globale per l'immagine della scacchiera
					global img_chessboard

					# L'immagine della scacchiera viene estratta
					img_chessboard = get_chessboard(img, chessboard_found, chessboard_corners)

					# Definizione nuove variabili globali in utilizzo
					global boxes_found
					global box_corners
					global coordinates

					# Ricerca delle case della scacchiera se queste non sono state mai trovate
					if boxes_found == False:
						# Ricerca della case all'interno della scacchiera
						boxes_found, box_corners = cv2.findChessboardCorners(pre_processing(img_chessboard), (7, 7),
																			 None)
						# Salvataggio delle coordinate all'interno di una struttura dati
						coordinates = get_final_coordinates(box_corners)

				# Salvataggio immagine per il thread

				if (chessboard_found and boxes_found):
					self.change_pixmap_signal.emit(img_chessboard)

		# Rilascio risorse allocate al termine delle operazioni
		webcam.release()

	# Quando il programma viene terminato
	def stop(self):
		self._run_flag = False
		self.wait()


"""Classe per la definizione dell'interfaccia del programma"""


class App(QWidget):
	def __init__(self):
		super().__init__()

		global isLoading
		global isAvailable
		isLoading = False

		"""Definizione della dimensione della finestra principale
		oltre alla definizione di una icona e di un nome per
		la stessa """
		self.setWindowTitle("Chess Computer System ")
		self.setObjectName("main_window")
		self.display_width = 1080
		self.display_height = 720
		self.setWindowIcon(QtGui.QIcon("horse.png"))

		"""Inserimento di un logo per riempire la interfacia"""
		self.logo = QLabel(self)
		self.logo.setGeometry(640,320,120,120)
		icon = QPixmap("horse.png")
		icon = icon.scaled(120, 120)
		self.logo.setPixmap(icon)


		"""Contorno per l'immagine della webcam nella home"""
		self.backgroundweb = QLabel(self)
		self.backgroundweb.setGeometry(40, 30, 400, 400)
		img_back = QPixmap("Assets/whiteback.png")
		img_back = img_back.scaled(400, 400)
		self.backgroundweb.setPixmap(img_back)

		"""Definizione di una finestra all'interno della quale 
		verrà visualizzata l'istanza relativa alla webcam , quindi 
		la scacchiera in live """
		self.image_label = QLabel(self)
		self.image_label.setGeometry(55, 45, 370, 370)
		self.image_label.setObjectName("LiveChessboard")
		self.image_label.setStyleSheet("border: 1px solid #202121;")

		"""Definzione di una finestra all'interno della quale verà 
		visualizzata la scacchiera elaborata a partire da un file svg"""
		self.widgetSvg = QSvgWidget(parent=self)
		self.widgetSvg.setGeometry(920, 30, 400, 400)

		"""Definizione di un pulsante per la modalità di analisi: 
		ho utilizzto questa modalità per lo più per testare le funzioni 
		e la funzionalità di alcuni strumenti, tra cui il riconoscimento 
		dei pezzi all'interno della scacchiera """
		self.solitario = QPushButton('Analisi partita', self)
		self.solitario.setToolTip('Gioca con un tuo amico!')
		self.solitario.setGeometry(600, 70, 200, 40)
		self.solitario.clicked.connect(self.on_click_solitario)

		"""Pulsante per inizizare una nuova partita in modalità singleplayer"""
		self.startBlack = QPushButton('Inizia partita', self)
		self.startBlack.setGeometry(600, 70, 200, 40)
		self.startBlack.clicked.connect(self.play_as_black)
		self.startBlack.hide()

		"""Pulsante per iniziare una nuova partita in modalità single player lato chiaro"""
		self.startWhite = QPushButton('Inizia partita', self)
		self.startWhite.setGeometry(600, 70, 200, 40)
		self.startWhite.clicked.connect(self.play_as_white)
		self.startWhite.hide()

		"""Definizione di un etichetta che rappresenterà la stringa delle 
		mosse eseguite : l'etichetta presenta dei contorni e un colore di 
		sfondo , oltre a delle ombre per rendere l'interfaccia più
		piacevole. La stampa delle stringhe all'interno della stessa parte 
		dall'alto a destra ( SetAligment ) """
		self.label = QLabel("Qui verranno mostrate le tue mosse", self)
		self.label.frameShadow()
		self.label.setGeometry(920, 450, 400, 200)
		self.label.setStyleSheet("""background-color: #eff4f7; border-radius:5px;
				border:1px solid #141010;font-family:Arial;
				font-size:12px;""")
		self.label.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignTop)
		self.label.setWordWrap(True)
		self.label.hide()



		"""Definizione di un pulsante per la modalità di gioco : 
		Bianco vs CPU """
		self.White = QPushButton('Gioca con il bianco', self)
		self.White.setToolTip('Clicca per giocare contro il Computer')
		self.White.setGeometry(600, 130, 200, 40)
		self.White.clicked.connect(self.start_as_white)

		"""Definizione di un pulsante per la modalità di gioco : 
		Nero  vs CPU """
		self.Black = QPushButton('Gioca con il nero', self)
		self.Black.setToolTip('Clicca per giocare contro il Computer')
		self.Black.setGeometry(600, 190, 200, 40)
		self.Black.clicked.connect(self.start_as_black)

		"""Pulsante per confermare l'esecuzione di una mossa """
		self.button = QPushButton('Prossima mossa', self)
		self.button.setToolTip("Clicca il tasto n per passare alla prossima mossa")
		self.button.setShortcut("n")
		self.button.setGeometry(600, 70, 200, 40)
		self.button.clicked.connect(self.on_click_next)
		self.button.hide()

		"""Pulsante per confermare l'esecuzione della mossa,
		utilizzato nella modalità di gioco Bianco Vs Cpu"""
		self.next_white = QPushButton('Prossima mossa', self)
		self.next_white.setToolTip("Clicca il tasto n per passare alla prossima mossa")
		self.next_white.setGeometry(600, 70, 200, 40)
		self.next_white.clicked.connect(self.on_click_next_white)
		self.next_white.hide()

		"""Pulsante per confermare l'esecuzione della mossa,
		utilizzato nella modalità di gioco Bianco Vs Cpu"""
		self.next_black = QPushButton('Prossima mossa', self)
		self.next_black.setToolTip("Clicca il tasto n per passare alla prossima mossa")
		self.next_black.setGeometry(600, 70, 200, 40)
		self.next_black.clicked.connect(self.on_click_next_black)
		self.next_black.hide()

		"""Pulsante per effettuare il reset della scacchiera
		e dello stato di gioco (Ad esempio quando si vuole 
		 iniziare una nuova partita) """
		self.buttonReset = QPushButton('Reset', self)
		self.buttonReset.setToolTip("Ricomincia la partita")
		self.buttonReset.setGeometry(600, 120, 200, 40)
		self.buttonReset.clicked.connect(self.on_click_reset)
		self.buttonReset.hide()

		"""Pulsante per rientrare al menu principale a partire
		dalle altre schermata di gioco """
		self.home = QPushButton("Torna alla home", self)
		self.home.setToolTip("Clicca per uscire")
		self.home.setGeometry(600, 170, 200, 40)
		self.home.clicked.connect(self.back_home)
		self.home.hide()

		"""Pulsante per effettuare la ricerca della scacchiera"""
		self.findChessboard = QPushButton('Ricerca Scacchiera', self)
		self.findChessboard.setToolTip("Localizza la scacchiera")
		self.findChessboard.setGeometry(600, 70, 200, 40)
		self.findChessboard.clicked.connect(self.search_chessboard)
		self.findChessboard.hide()

		"""Combo box per la scelta del livello"""
		self.slider = QComboBox(self)
		self.slider.setGeometry(600, 120, 200, 40)
		self.slider.addItem("Difficoltà")
		self.slider.addItem("Principiante")
		self.slider.addItem("Esperto")
		self.slider.addItem("Maestro")
		self.slider.addItem("Gran Maestro")
		self.slider.hide()
		self.slider.setCurrentIndex(0)
		self.slider.setToolTip('Seleziona il livello')



		"""Pulsante per le impostazioni della scacchiera """
		self.setting = QPushButton("Impostazioni ", self)
		self.setting.setToolTip("Clicca per aprire le impostazioni")
		self.setting.setGeometry(600, 250, 200, 40)
		self.setting.clicked.connect(self.open_settings)

		"""Pulsante per wizard creazione set scacchiera"""
		self.wizard = QPushButton("Aggiungi scacchiera", self)
		self.wizard.setToolTip("Clicca per creare poter giocare con la tua scacchiera")
		self.wizard.setGeometry(600, 120, 200, 40)
		self.wizard.clicked.connect(self.start_wizard)
		self.wizard.hide()

		"""Pulsante per procedere nel wizard"""
		self.next_wizard = QPushButton("Successivo", self)
		self.next_wizard.setGeometry(600, 70, 200, 40)
		self.next_wizard.clicked.connect(self.next_wizard_action)
		self.next_wizard.hide()

		"""Loading Bar per creazione del modello"""
		self.progressBar = QProgressBar(self)
		self.progressBar.setGeometry(600, 70, 200, 40)
		self.progressBar.setProperty("value", 0)
		self.progressBar.setTextVisible(True)
		self.progressBar.hide()

		"""Thread per la creazione del modello"""
		self.myLongTask = TaskThread()
		self.myLongTask.taskFinished.connect(self.onFinished)

		"""Codifica della scacchiera formato svg per la lettura 
		e scrittura all'interno della finestra preposta """
		self.chessboardSvg = chess.svg.board(chessboard).encode("UTF-8")
		self.widgetSvg.load(self.chessboardSvg)

		self.radio_label = QLabel("Seleziona scacchiera",self)
		self.radio_label.setGeometry(600,170,200,40)
		self.radio_label.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignTop)
		self.radio_label.setObjectName("RadioLabel")
		self.radio_label.hide()

		self.radioDefault = QRadioButton("Default",self)
		self.radioDefault.setGeometry(610,177,200,40)
		self.radioDefault.setChecked(True)
		self.radioDefault.hide()

		self.radioPersonal = QRadioButton("Personale",self)
		self.radioPersonal.setGeometry(690,177,200,40)
		self.radioPersonal.setCheckable(isAvailable)
		self.radioPersonal.hide()

		self.radioDefault.toggled.connect(self.onClicked)




	def onClicked(self):
		global chessboard_type
		if(self.radioDefault.isChecked()):
			chessboard_type = 0
			set_model(chessboard_type)
			return
		else:
			chessboard_type = 1
			set_model(chessboard_type)



	def onFinished(self):
		# Stop the pulsation
		self.progressBar.setRange(0, 100)
		self.myLongTask.stop()
		isPersonalAvailable()



		self.msg = QMessageBox()
		self.msg.setWindowTitle("CONGRATULAZIONI!")
		self.msg.setText("Puoi ora utilizzare il sistema con la tua scacchiera")
		self.msg.setIcon(QMessageBox.Information)
		self.msg.setStandardButtons(QMessageBox.Ok)
		self.msg.setWindowFlag(Qt.FramelessWindowHint)
		self.msg.show()

		self.msg.buttonClicked.connect(self.back_home)



	"""Funzione che permette di tornare al menu iniziale 
	a partire da una schermata di gioco : vengono resettati i pulsanti 
	presenti nella home e viene resettata la scacchiera"""
	@pyqtSlot()
	def back_home(self):

		global isWhiteTurn
		global isBlackTurn
		global oldblack
		global oldwhite
		global chessboard

		isWhiteTurn = True
		isBlackTurn = False

		oldblack = setBlack()
		oldwhite = setWhite()

		"""Resetto lo scacchiera sulla sinistra """
		chessboard = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")
		self.updateChessboard(chessboard,0)
		self.label.setText("")

		"""Resetto la scacchiera sulla destra ( bordi numerati ) """
		img_back = QPixmap("Assets/whiteback.png")
		img_back = img_back.scaled(400, 400)
		self.backgroundweb.setPixmap(img_back)

		self.chessboardSvg = chess.svg.board(chessboard,orientation=chess.WHITE).encode("UTF-8")
		self.widgetSvg.load(self.chessboardSvg)

		# Aggiorna pulsanti
		self.solitario.show()
		self.White.show()
		self.Black.show()
		self.next_white.hide()
		self.buttonReset.hide()
		self.label.hide()
		self.next_black.hide()
		self.home.hide()
		self.slider.hide()
		self.setting.show()
		self.findChessboard.hide()
		self.progressBar.hide()
		self.wizard.hide()
		self.radio_label.hide()
		self.radioPersonal.hide()
		self.radioDefault.hide()
		self.button.hide()

	"""Mostra le impostazioni """
	@pyqtSlot()
	def open_settings(self):
		self.findChessboard.show()
		self.solitario.hide()
		self.White.hide()
		self.Black.hide()
		self.home.show()
		self.setting.hide()
		self.wizard.show()
		self.radioDefault.show()
		self.radioPersonal.show()
		self.radio_label.show()
		self.home.setGeometry(600,250,200,40)

	@pyqtSlot()
	def start_wizard(self):
		self.label.show()
		self.wizard.hide()
		self.home.hide()
		self.findChessboard.hide()
		self.next_wizard.show()
		self.radioPersonal.hide()
		self.radioDefault.hide()
		self.radio_label.hide()
		self.label.setText("Inserisci i pezzi sulla scacchiera , cosi come sono indicati all"
						   "interno dell'immagine qui sopra , quindi clicca sul pulsante "
						   "'Successivo'.")

		chessboard = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
		self.chessboardSvg = chess.svg.board(chessboard,orientation=chess.WHITE).encode("UTF-8")
		self.widgetSvg.load(self.chessboardSvg)

	@pyqtSlot()
	def next_wizard_action(self):
		global step


		if(step == 0 ):
			boxes = boxes_matrix(img_chessboard, coordinates)
			create_chessboard_set(boxes,step)

			"""Prossimo schema per i pezzi sulla scacchiera"""
			chessboard = chess.Board("rnbqkbnr/8/pppppppp/8/8/PPPPPPPP/8/RNBQKBNR w - - 0 1")
			self.chessboardSvg = chess.svg.board(chessboard, orientation=chess.WHITE).encode("UTF-8")
			self.widgetSvg.load(self.chessboardSvg)


		if(step == 1):
			boxes = boxes_matrix(img_chessboard, coordinates)
			create_chessboard_set(boxes,step)
			"""Prossimo schema per i pezzi sulla scacchiera"""
			chessboard = chess.Board("8/rnbqkbnr/pppppppp/8/8/PPPPPPPP/RNBQKBNR/8 w - - 0 1")
			self.chessboardSvg = chess.svg.board(chessboard, orientation=chess.WHITE).encode("UTF-8")
			self.widgetSvg.load(self.chessboardSvg)



		if(step == 2 ):
			boxes = boxes_matrix(img_chessboard, coordinates)
			create_chessboard_set(boxes,step)

			self.next_wizard.hide()
			self.progressBar.show()
			self.progressBar.setRange(0, 100)
			self.myLongTask.start()

			"""Prossimo schema per i pezzi sulla scacchiera"""
			chessboard = chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")
			self.chessboardSvg = chess.svg.board(chessboard, orientation=chess.WHITE).encode("UTF-8")
			self.widgetSvg.load(self.chessboardSvg)
			step = 0
			return


		step = step + 1


	"""Funzione che permette di resettare i flag per la scacchiera :
	resettandoli riparte la ricerca della stessa """

	@pyqtSlot()
	def search_chessboard(self):

		global chessboard_found, boxes_found

		chessboard_found = False
		boxes_found = False




	"""Funzione che permette di iniziare la partita giocando con il 
	lato del bianco """

	@pyqtSlot()
	def play_as_white(self):
		global chessboard
		self.setLevel()

		# Estrazione delle case della matrice
		if boxes_found:
			boxes = boxes_matrix(img_chessboard, coordinates)

		# Creazione matrice posizionale
		matrix = find_pieces(boxes, "all")

		if isStart(matrix):
			# La scacchiera è nel suo stato iniziale
			chessboard = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

			# Aggiorna pulsanti
			self.solitario.hide()
			self.White.hide()
			self.Black.hide()
			self.next_white.show()
			self.buttonReset.show()
			self.label.show()
			self.findChessboard.hide()
			self.home.show()
			self.slider.hide()
			self.setting.hide()

			# Aggiorno la scacchiera a schermo
			self.updateChessboard(chessboard,0)

		elif isEmpty(matrix):
			self.msg = QMessageBox()
			self.msg.setWindowTitle("OPS...")
			self.msg.setText("Inserisci i pezzi sulla scacchiera prima di cominciare.")
			self.msg.setIcon(QMessageBox.Information)
			self.msg.setStandardButtons(QMessageBox.Ok)
			self.msg.setWindowFlag(Qt.FramelessWindowHint)
			self.msg.show()

	@pyqtSlot()
	def start_as_white(self):
		# Aggiorna pulsanti
		self.solitario.hide()
		self.White.hide()
		self.Black.hide()
		self.startWhite.show()
		self.label.show()
		self.findChessboard.hide()
		self.home.show()
		self.slider.show()
		self.setting.hide()

	@pyqtSlot()
	def start_as_black(self):

		"""Modifica della scacchiera ,
		posizionata al contrario, in modo tale da
		 favorire la scacchiera rispeotto all'utente"""
		img_back = QPixmap("Assets/blackback.png")
		img_back = img_back.scaled(400, 400)
		self.backgroundweb.setPixmap(img_back)

		self.chessboardSvg = chess.svg.board(chessboard,orientation=chess.BLACK).encode("UTF-8")
		self.widgetSvg.load(self.chessboardSvg)

		# Aggiorna pulsanti
		self.solitario.hide()
		self.White.hide()
		self.Black.hide()
		self.startBlack.show()
		self.label.show()
		self.findChessboard.hide()
		self.home.show()
		self.slider.show()
		self.setting.hide()

	@pyqtSlot()
	def play_as_black(self):

		global chessboard

		self.chessboardSvg = chess.svg.board(chessboard, orientation=chess.BLACK).encode("UTF-8")
		self.widgetSvg.load(self.chessboardSvg)

		self.setLevel()



		# Estrazione delle case della matrice
		if boxes_found:
			boxes = boxes_matrix(img_chessboard, coordinates)

		# Creazione matrice posizionale
		matrix = find_pieces(boxes, "all")

		if isStart(matrix):
			# La scacchiera è nel suo stato iniziale
			chessboard = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

			# Aggiorna pulsanti
			self.solitario.hide()
			self.White.hide()
			self.Black.hide()
			self.next_black.show()
			self.buttonReset.show()
			self.label.show()
			self.findChessboard.hide()
			self.home.show()
			self.slider.hide()
			self.setting.hide()



			"""Prima mossa spetta al computer """
			fen = chessboard.fen()
			stockfish.set_fen_position(fen)

			best_move = stockfish.get_best_move()
			chessboard.push_uci(best_move)

			# Aggiorna mosse
			self.on_update_moves(chessboard)

			# Aggiorna la scacchiera a schermo
			self.updateChessboard(chessboard,1)

		elif isEmpty(matrix):
			self.msg = QMessageBox()
			self.msg.setWindowTitle("OPS...")
			self.msg.setText("Inserisci i pezzi sulla scacchiera prima di cominciare.")
			self.msg.setIcon(QMessageBox.Information)
			self.msg.setStandardButtons(QMessageBox.Ok)
			self.msg.setWindowFlag(Qt.FramelessWindowHint)
			self.msg.show()

	"""Funzione che permette di giocare una modalità
	solitaria o analizzare una partita in corso , 
	salvare le mosse ..."""

	@pyqtSlot()
	def on_click_solitario(self):

		global chessboard

		# Estrazione delle case della matrice
		if boxes_found:
			boxes = boxes_matrix(img_chessboard, coordinates)

		# Creazione matrice posizionale
		matrix = find_pieces(boxes, "all")

		if isStart(matrix):
			# La scacchiera è nel suo stato iniziale
			chessboard = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
			# Aggiorna pulsanti
			self.solitario.hide()
			self.button.show()
			self.buttonReset.show()
			self.label.show()
			self.White.hide()
			self.Black.hide()
			self.findChessboard.hide()
			self.setting.hide()
			self.home.show()

			# Aggiorno la scacchiera a schermo
			self.updateChessboard(chessboard,0)

		elif isEmpty(matrix):
			self.msg = QMessageBox()
			self.msg.setWindowTitle("OPS...")
			self.msg.setText("Inserisci i pezzi sulla scacchiera prima di cominciare.")
			self.msg.setIcon(QMessageBox.Information)
			self.msg.setStandardButtons(QMessageBox.Ok)
			self.msg.setWindowFlag(Qt.FramelessWindowHint)
			self.msg.show()

	"""Permette di resettare il gioco """

	@pyqtSlot()
	def on_click_reset(self):
		global isWhiteTurn
		global isBlackTurn
		global oldblack
		global oldwhite
		global chessboard

		isWhiteTurn = True
		isBlackTurn = False

		oldblack = setBlack()
		oldwhite = setWhite()

		chessboard = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
		self.updateChessboard(chessboard,0)
		self.label.setText("")

	"""Permette di passare da una mossa all'altra ,
	all'interno della modalità di gioco 
	bianco vs cpu """

	@pyqtSlot()
	def on_click_next_white(self):
		# Variabile globale da utilizzare
		global oldwhite

		# Estrazione delle case della matrice
		if boxes_found:
			boxes = boxes_matrix(img_chessboard, coordinates)

		# Estrazione matrice posizionale bianca
		matrix = find_pieces(boxes, "white")
		# Elaborazione mossa del bianco
		try:
			move = get_move(oldwhite, matrix, chessboard)
			chessboard.push_uci(move)
		except:
			self.msg = QMessageBox()
			self.msg.setWindowTitle("ATTENZIONE!")
			self.msg.setText("Mossa non valida")
			self.msg.setIcon(QMessageBox.Warning)
			self.msg.setStandardButtons(QMessageBox.Ok)
			self.msg.setWindowFlag(Qt.FramelessWindowHint)
			self.msg.show()
			return

		# Vittoria per il bianco
		if (chessboard.is_checkmate()):
			self.msg = QMessageBox()
			self.msg.setWindowTitle("CONGRATULAZIONI ! ")
			self.msg.setText("Hai vinto la partita")
			self.msg.setIcon(QMessageBox.Information)
			self.msg.setStandardButtons(QMessageBox.Ok)
			self.msg.setWindowFlag(Qt.FramelessWindowHint)
			self.msg.show()
			return

		if (chessboard.is_repetition(3) or chessboard.is_insufficient_material() or chessboard.is_stalemate()):
			self.msg = QMessageBox()
			self.msg.setWindowTitle("PARTITA PATTA!")
			self.msg.setText("Partita patta : non è possibile stabilire un vincitore tra i giocatori")
			self.msg.setIcon(QMessageBox.Information)
			self.msg.setStandardButtons(QMessageBox.Ok)
			self.msg.setWindowFlag(Qt.FramelessWindowHint)
			self.msg.show()
			return



		# Aggiorna la scacchiera a schermo
		self.updateChessboard(chessboard,0)
		# Sovrascrivere variabile dello stato precedente con il nuovo stato
		oldwhite = get_old_matrix(chessboard,"white")

		fen = chessboard.fen()
		stockfish.set_fen_position(fen)

		best_move = stockfish.get_best_move()
		chessboard.push_uci(best_move)

		oldwhite = computer_black_move(best_move, oldwhite)

		self.on_update_moves(chessboard)

		# Aggiorna la scacchiera a schermo
		self.updateChessboard(chessboard,0)

		# Vittoria per il nero
		if (chessboard.is_checkmate()):
			self.msg = QMessageBox()
			self.msg.setWindowTitle("HAI PERSO !")
			self.msg.setText("Il computer ti ha battuto")
			self.msg.setIcon(QMessageBox.Information)
			self.msg.setStandardButtons(QMessageBox.Ok)
			self.msg.setWindowFlag(Qt.FramelessWindowHint)
			self.msg.show()
			return


		if (chessboard.is_repetition(3) or chessboard.is_insufficient_material() or chessboard.is_stalemate()):
			self.msg = QMessageBox()
			self.msg.setWindowTitle("PARTITA PATTA!")
			self.msg.setText("Partita patta : non è possibile stabilire un vincitore tra i giocatori")
			self.msg.setIcon(QMessageBox.Information)
			self.msg.setStandardButtons(QMessageBox.Ok)
			self.msg.setWindowFlag(Qt.FramelessWindowHint)
			self.msg.show()
			return

	@pyqtSlot()
	def on_click_next_black(self):
		# Variabile globale da utilizzare
		global oldblack

		# Estrazione delle case della matrice
		if boxes_found:
			boxes = boxes_matrix(img_chessboard, coordinates)

		# Estrazione matrice posizionale bianca
		matrix = find_pieces(boxes, "black")

		matrix = rotate_matrix(matrix)

		print_positional_matrix(matrix)

		# Elaborazione mossa del nero
		try:
			move = get_move(oldblack, matrix, chessboard)
			chessboard.push_uci(move)
		except:
			self.msg = QMessageBox()
			self.msg.setWindowTitle("ATTENZIONE!")
			self.msg.setText("Mossa non valida")
			self.msg.setIcon(QMessageBox.Warning)
			self.msg.setStandardButtons(QMessageBox.Ok)
			self.msg.setWindowFlag(Qt.FramelessWindowHint)
			self.msg.show()
			return



		# Aggiorna la scacchiera a schermo
		self.updateChessboard(chessboard,1)
		# Sovrascrivere variabile dello stato precedente con il nuovo stato
		oldblack = get_old_matrix(chessboard,"black")

		# Vittoria per il nero
		if (chessboard.is_checkmate()):
			self.msg = QMessageBox()
			self.msg.setWindowTitle("CONGRATULAZIONI ! ")
			self.msg.setText("Hai vinto la partita")
			self.msg.setIcon(QMessageBox.Information)
			self.msg.setStandardButtons(QMessageBox.Ok)
			self.msg.setWindowFlag(Qt.FramelessWindowHint)
			self.msg.show()
			return

		if (chessboard.is_repetition(3) or chessboard.is_insufficient_material() or chessboard.is_stalemate()):
			self.msg = QMessageBox()
			self.msg.setWindowTitle("PARTITA PATTA!")
			self.msg.setText("Partita patta : non è possibile stabilire un vincitore tra i giocatori")
			self.msg.setIcon(QMessageBox.Information)
			self.msg.setStandardButtons(QMessageBox.Ok)
			self.msg.setWindowFlag(Qt.FramelessWindowHint)
			self.msg.show()
			return



		fen = chessboard.fen()
		stockfish.set_fen_position(fen)

		best_move = stockfish.get_best_move()
		chessboard.push_uci(best_move)

		"""Nel caso il computer prende un pezzo del nero"""
		oldblack = computer_black_move(best_move, oldblack)

		# Aggiorna mosse
		self.on_update_moves(chessboard)

		# Aggiorna la scacchiera a schermo
		self.updateChessboard(chessboard,1)


		# Vittoria per il bianco
		if (chessboard.is_checkmate()):
			self.msg = QMessageBox()
			self.msg.setWindowTitle("HAI PERSO !")
			self.msg.setText("Il computer ti ha battuto")
			self.msg.setIcon(QMessageBox.Information)
			self.msg.setStandardButtons(QMessageBox.Ok)
			self.msg.setWindowFlag(Qt.FramelessWindowHint)
			self.msg.show()
			return


		if (chessboard.is_repetition(3) or chessboard.is_insufficient_material() or chessboard.is_stalemate()):
			self.msg = QMessageBox()
			self.msg.setWindowTitle("PARTITA PATTA!")
			self.msg.setText("Partita patta : non è possibile stabilire un vincitore tra i giocatori")
			self.msg.setIcon(QMessageBox.Information)
			self.msg.setStandardButtons(QMessageBox.Ok)
			self.msg.setWindowFlag(Qt.FramelessWindowHint)
			self.msg.show()
			return

	"""Aggiorna le mosse all'interno del label"""

	def on_update_moves(self, chessboard):

		stack_moves = ""
		index_move = 2

		for move in chessboard.move_stack:
			stack_moves = stack_moves + str(int(index_move / 2)) + "." + chessboard.uci(move) + " "
			index_move = index_move + 1

		self.label.setText(stack_moves)

	def setLevel(self):
		i = self.slider.currentIndex()

		if i == 1 :
			stockfish.set_elo_rating(elo_rating=1000)
		elif i == 2 :
			stockfish.set_elo_rating(elo_rating=1500)
		elif i == 3:
			stockfish.set_elo_rating(elo_rating=2000)
		elif i == 4:
			stockfish.set_elo_rating(elo_rating=2500)
		else : pass


	"""Prossima mossa in modalità analisi"""
	@pyqtSlot()
	def on_click_next(self):

		# Variabile globale da utilizzare
		global isWhiteTurn
		global isBlackTurn
		global oldblack
		global oldwhite
		global chessboard

		print("OLDWHITE")
		print_positional_matrix(oldwhite)

		print("OLDBLACK")
		print_positional_matrix(oldblack)

		# Estrazione delle case della matrice
		if boxes_found:
			boxes = boxes_matrix(img_chessboard, coordinates)

		# Se è il turno del bianco
		if isWhiteTurn:

			# Estrazione matrice posizionale bianca
			matrix = find_pieces(boxes, "white")

			try:
				# Elaborazione mossa del bianco
				move = get_move_single(oldwhite, matrix, chessboard, oldblack)
				chessboard.push_uci(move)
			except:
				self.msg = QMessageBox()
				self.msg.setWindowTitle("ATTENZIONE!")
				self.msg.setText("Mossa non valida")
				self.msg.setIcon(QMessageBox.Warning)
				self.msg.setStandardButtons(QMessageBox.Ok)
				self.msg.setWindowFlag(Qt.FramelessWindowHint)
				self.msg.show()
				return

			# Gestione pedone mangiato
			oldblack = get_old_matrix(chessboard,"black")

			# Vittoria per il bianco
			if (chessboard.is_checkmate()):
				self.msg = QMessageBox()
				self.msg.setWindowTitle("CONGRATULAZIONI ! ")
				self.msg.setText("Il giocatore bianco ha vinto la partita")
				self.msg.setIcon(QMessageBox.Information)
				self.msg.setStandardButtons(QMessageBox.Ok)
				self.msg.setWindowFlag(Qt.FramelessWindowHint)
				self.msg.show()

			if (chessboard.is_repetition(3) or chessboard.is_insufficient_material() or chessboard.is_stalemate()):
				self.msg = QMessageBox()
				self.msg.setWindowTitle("PARTITA PATTA!")
				self.msg.setText("Partita patta : non è possibile stabilire un vincitore tra i giocatori")
				self.msg.setIcon(QMessageBox.Information)
				self.msg.setStandardButtons(QMessageBox.Ok)
				self.msg.setWindowFlag(Qt.FramelessWindowHint)
				self.msg.show()
				return



			self.on_update_moves(chessboard)

			# Gestione dei turni
			isWhiteTurn = False
			isBlackTurn = True
			# Aggiorna la scacchiera a schermo
			self.updateChessboard(chessboard,0)
			# Sovrascrivere variabile dello stato precedente con il nuovo stato
			oldwhite = get_old_matrix(chessboard,"white")


			return

		# Se è il turno del nero
		if isBlackTurn:
			# Estrazione matrice posizionale nera
			matrix = find_pieces(boxes, "black")

			try:
				# Elaborazione mossa del bianco
				move = get_move_single(oldblack, matrix, chessboard, oldwhite)
				chessboard.push_uci(move)
			except:
				self.msg = QMessageBox()
				self.msg.setWindowTitle("ATTENZIONE!")
				self.msg.setText("Mossa non valida")
				self.msg.setIcon(QMessageBox.Warning)
				self.msg.setStandardButtons(QMessageBox.Ok)
				self.msg.setWindowFlag(Qt.FramelessWindowHint)
				self.msg.show()
				return

			# Gestione pedone mangiato
			oldwhite = get_old_matrix(chessboard,"white")

			# Vittoria per il bianco
			if (chessboard.is_checkmate()):
				self.msg = QMessageBox()
				self.msg.setWindowTitle("CONGRATULAZIONI ! ")
				self.msg.setText("Il giocatore nero ha vinto la partita")
				self.msg.setIcon(QMessageBox.Information)
				self.msg.setStandardButtons(QMessageBox.Ok)
				self.msg.setWindowFlag(Qt.FramelessWindowHint)
				self.msg.show()

			if (chessboard.is_repetition(3) or chessboard.is_insufficient_material() or chessboard.is_stalemate()):
				self.msg = QMessageBox()
				self.msg.setWindowTitle("PARTITA PATTA!")
				self.msg.setText("Partita patta : non è possibile stabilire un vincitore tra i giocatori")
				self.msg.setIcon(QMessageBox.Information)
				self.msg.setStandardButtons(QMessageBox.Ok)
				self.msg.setWindowFlag(Qt.FramelessWindowHint)
				self.msg.show()
				return




			self.on_update_moves(chessboard)

			# Passa il turno
			isWhiteTurn = True
			isBlackTurn = False
			# Aggiorna la scacchiera a schermo
			self.updateChessboard(chessboard,0)
			# Sovrascrivere variabile dello stato precedente con il nuovo stato
			oldblack = get_old_matrix(chessboard,"black")
			return

	"""Aggiorna la scacchiera a schermo"""

	@pyqtSlot()
	def updateChessboard(self, chessboard,flag):

		if(flag == 0):
			self.chessboardSvg = chess.svg.board(chessboard).encode("UTF-8")
			self.widgetSvg.load(self.chessboardSvg)

		if(flag == 1 ):
			self.chessboardSvg = chess.svg.board(chessboard, orientation=chess.BLACK).encode("UTF-8")
			self.widgetSvg.load(self.chessboardSvg)


	"""Funzione per aggiornare l'immagine 
	catturata dalla webcam all'interno 
	della finestra presente nell'interfaccia """

	@pyqtSlot(np.ndarray)
	def update_image(self, cv_img):
		"""Updates the image_label with a new opencv image"""
		qt_img = self.convert_cv_qt(cv_img)
		self.image_label.setPixmap(qt_img)

	"""Funzione per convertire l'immagine gestita 
	tramite OpenCV in QpixMap """

	def convert_cv_qt(self, cv_img):
		# Rotazione necessaria per avere i pezzi bianchi sul proprio lato
		cv_img = cv2.rotate(cv_img, cv2.ROTATE_180)
		# Conversione in QImage
		rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		h, w, ch = rgb_image.shape
		bytes_per_line = ch * w
		p = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
		return QPixmap.fromImage(p)

def isPersonalAvailable():
	global isAvailable

	if Path("my_keras.h5").is_file():
		isAvailable = True




"""Definizione della schermata di caricamento"""


class SplashScreen(QWidget):
	def __init__(self):
		super().__init__()

		# Impostazione finestra principale
		self.setWindowTitle('Spash Screen')
		self.setFixedSize(1100, 500)

		# Rimozione barra titolo
		self.setWindowFlag(Qt.FramelessWindowHint)

		# Impostazione traslucenza
		self.setAttribute(Qt.WA_TranslucentBackground)

		# Contatore per il caricamento
		self.counter = 0
		self.n = 200

		# Inizializzazione interfaccia
		self.initUI()

		"""Utilizzo di un thread per la gestione del video 
		catturato dalla webcam che ritrae la scacchiera : è 
		necessario utilizzare un thread per non bloccare 
		l'interfaccia ( INR ) """
		self.thread = VideoThread()
		self.thread.change_pixmap_signal.connect(self.update_image)
		self.thread.start()

		# Gestione timer per il caricamento
		self.timer = QTimer()
		self.timer.timeout.connect(self.loading)
		self.timer.start(30)

	"""Funzione per la definizione degli elementi 
	presenti all'interno della interfaccia 
	della splash screen """

	def initUI(self):

		isPersonalAvailable()


		# Definizione tipologia di layout
		layout = QVBoxLayout()

		# Impostazione layout
		self.setLayout(layout)

		"""Definizione frame contenente la barra 
		di caricamento : il nome viene utilizzato all'interno
		di stylesheet , quindi all'interno del codice CSS """
		self.frame = QFrame()
		self.frame.setObjectName('FrameLoader')
		layout.addWidget(self.frame)

		# Definizione titolo della finestra
		self.labelTitle = QLabel(self.frame)
		self.labelTitle.setObjectName('LabelTitle')

		# Label centrale
		self.labelTitle.resize(self.width() - 10, 150)
		self.labelTitle.move(0, 40)  # x, y
		self.labelTitle.setText('Chess Computer Vision System')
		self.labelTitle.setAlignment(Qt.AlignCenter)

		# Sottotitolo : cambia durante il caricamento
		self.labelDescription = QLabel(self.frame)
		self.labelDescription.resize(self.width() - 10, 50)
		self.labelDescription.move(0, self.labelTitle.height())
		self.labelDescription.setObjectName('LabelDesc')
		self.labelDescription.setText('<strong>Gestione intelligenza artificiale</strong>')
		self.labelDescription.setAlignment(Qt.AlignCenter)

		# ProgressBar per restituire un feedback per il caricamento
		self.progressBar = QProgressBar(self.frame)
		self.progressBar.resize(self.width() - 200 - 10, 50)
		self.progressBar.move(100, self.labelDescription.y() + 130)
		self.progressBar.setAlignment(Qt.AlignCenter)
		self.progressBar.setFormat('%p%')
		self.progressBar.setTextVisible(True)
		self.progressBar.setRange(0, self.n)
		self.progressBar.setValue(20)

		# Scritta che indica il processo in corso
		self.labelLoading = QLabel(self.frame)
		self.labelLoading.resize(self.width() - 10, 50)
		self.labelLoading.move(0, self.progressBar.y() + 70)
		self.labelLoading.setObjectName('LabelLoading')
		self.labelLoading.setAlignment(Qt.AlignCenter)
		self.labelLoading.setText('caricamento...')

	"""Come sopra , la gestione del thread è stata spostata 
	all'interno della splash screen , in modo tale da poter 
	caricare i contenuti necessari durante il caricamento """

	@pyqtSlot(np.ndarray)
	def update_image(self, cv_img):
		"""Updates the image_label with a new opencv image"""
		qt_img = self.convert_cv_qt(cv_img)
		if (not isLoading):
			self.myApp.image_label.setPixmap(qt_img)



	def convert_cv_qt(self, cv_img):
		"""Convert from an opencv image to QPixmap"""
		cv_img = cv2.rotate(cv_img, cv2.ROTATE_180)
		rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		h, w, ch = rgb_image.shape
		bytes_per_line = ch * w
		p = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
		return QPixmap.fromImage(p)

	def closeEvent(self, event):
		if (not isLoading):
			self.thread.stop()
			event.accept()

	"""Funzione per l'aggiornamento della barra 
	di caricamento e i vari sottotitoli indicanti 
	il processo in corso """

	def loading(self):

		self.progressBar.setValue(self.counter)

		if self.counter == int(self.n * 0.3):
			self.labelDescription.setText('<strong>Gestione visione artificiale</strong>')
		elif self.counter == int(self.n * 0.6):
			self.labelDescription.setText('<strong>Caricamento motore di scacchi</strong>')
		elif self.counter >= self.n:
			self.timer.stop()
			self.close()

			self.myApp = App()
			global refApp
			refApp= self.myApp
			self.myApp.show()

		self.counter += 1


"""Main dell'applicazione"""
if __name__ == "__main__":
	app = QApplication(sys.argv)
	app.setStyleSheet('''
			#LabelTitle {
				font-size: 60px;
				color: #93deed;
			}

			#LabelDesc {
				font-size: 30px;
				color: #c2ced1;
			}
			
			QComboBox {
				background-color:#8b9294;
				border-radius:8px;
				border:2px solid #141010;
				color:#000000;
				font-family:Arial;
				font-size:21px;
				text-align-last:center;

			}
			
			
			QComboBox::drop-down{
				border: 0px; 	
			}

			#LabelLoading {
				font-size: 30px;
				color: #e8e8eb;
			}

			#FrameLoader {
				background-color: #2F4454;
				color: rgb(220, 220, 220);
			}

			QProgressBar {
				background-color: #DA7B93;
				color: rgb(200, 200, 200);
				border-style: none;
				border-radius: 10px;
				text-align: center;
				font-size: 30px;
			}

			QProgressBar::chunk {
				border-radius: 10px;
				background-color: qlineargradient(spread:pad x1:0, x2:1, y1:0.511364, y2:0.523, stop:0 #1C3334, stop:1 #376E6F);
			}
			
			
			#main_window{
				background-color : #2f4455;					
			}
				
			QPushButton{
				background:linear-gradient(to bottom, #8b9294 5%, #70787a 100%);
				background-color:#8b9294;
				border-radius:8px;
				border:2px solid #141010;
				display:inline-block;
				cursor:pointer;
				color:#ffffff;
				font-family:Arial;
				font-size:21px;
				text-decoration:none;
				text-shadow:0px 1px 0px #000000;
			}
			
			#RadioLabel{
				background:linear-gradient(to bottom, #8b9294 5%, #70787a 100%);
				background-color:#8b9294;
				border-radius:8px;
				border:2px solid #141010;
				cursor:pointer;
				color:#ffffff;
				font-family:Arial;
				font-size:12px;
				text-decoration:none;
				text-shadow:0px 1px 0px #000000;
			}
			
			QPushButton:hover {
				background:linear-gradient(to bottom, #70787a 5%, #8b9294 100%);
				background-color:#70787a;
			}
			
			QPushButton:active {
				position:relative;
				top:1px;
			}
			
			
			#LiveChessboard {
				border : 3px gray;
			}
			
			QMessageBox{
				background-color: #2f4455;
				color:#ffffff;
				border : 2px solid black ; 
				border-radius: 10px;
				text-align: center;
				font-size: 15px;
				font-family:Arial;
			}
			
		''')
	splash = SplashScreen()
	splash.show()

	app.setStyle('Fusion')

	try:
		sys.exit(app.exec_())
	except SystemExit:
		print('Closing Window...')
