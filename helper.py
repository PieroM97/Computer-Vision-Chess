from PyQt5.QtGui import QPixmap

class Helper :

    helper_img = None
    helper_msg = ""

    def __init__(self):
        self.helper_img = QPixmap("Assets/chessAvatar.png")
        self.helper_img = self.helper_img.scaled(150, 150)
        self.helper_msg = "Ciao , ricorda di preparare la scacchiera prima di iniziare."

    def set_loading(self):
        self.helper_img = QPixmap("Assets/avatarthinking.png")
        self.helper_img = self.helper_img.scaled(150, 150)

        self.helper_msg = "Sto analizzando la scacchiera ... "

    def get_helper(self):
         return self.helper_img

    def get_message(self):
        return self.helper_msg

