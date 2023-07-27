from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel



# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account has not been activated by the Admin🛑🤚')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})

def UserDetect(request):
    import tkinter
    from PIL import Image
    from tkinter import filedialog
    import cv2 as cv
    from .AlgoProcess.frames import Frames
    from .AlgoProcess.displayTumor import DisplayTumor
    from .AlgoProcess.predictTumor import predictTumor




    class Gui:
        MainWindow = 0
        listOfWinFrame = list()
        FirstFrame = object()
        val = 0
        fileName = 0
        DT = object()

        wHeight = 700
        wWidth = 1180

        def __init__(self):
            global MainWindow
            MainWindow = tkinter.Tk()
            MainWindow.geometry('1200x720')
            MainWindow.resizable(width=False, height=False)

            self.DT = DisplayTumor()

            self.fileName = tkinter.StringVar()

            self.FirstFrame = Frames(self, MainWindow, self.wWidth, self.wHeight, 0, 0)
            self.FirstFrame.btnView['state'] = 'disable'

            self.listOfWinFrame.append(self.FirstFrame)

            WindowLabel = tkinter.Label(self.FirstFrame.getFrames(), text="Brain Tumor Detection", height=1, width=40)
            WindowLabel.place(x=320, y=30)
            WindowLabel.configure(background="White", font=("Comic Sans MS", 16, "bold"))

            self.val = tkinter.IntVar()
            RB1 = tkinter.Radiobutton(self.FirstFrame.getFrames(), text="Detect Tumor", variable=self.val,
                                    value=1, command=self.check)
            RB1.place(x=250, y=200)
            RB2 = tkinter.Radiobutton(self.FirstFrame.getFrames(), text="View Tumor Region",
                                    variable=self.val, value=2, command=self.check)
            RB2.place(x=250, y=250)

            browseBtn = tkinter.Button(self.FirstFrame.getFrames(), text="Browse", width=8, command=self.browseWindow)
            browseBtn.place(x=800, y=550)

            MainWindow.mainloop()

        def getListOfWinFrame(self):
            return self.listOfWinFrame

        def browseWindow(self):
            global mriImage
            FILEOPENOPTIONS = dict(defaultextension='*.*',
                                filetypes=[('jpg', '*.jpg'), ('png', '*.png'), ('jpeg', '*.jpeg'), ('All Files', '*.*')])
            self.fileName = filedialog.askopenfilename(**FILEOPENOPTIONS)
            image = Image.open(self.fileName)
            imageName = str(self.fileName)
            mriImage = cv.imread(imageName, 1)
            self.listOfWinFrame[0].readImage(image)
            self.listOfWinFrame[0].displayImage()
            self.DT.readImage(image)

        def check(self):
            global mriImage
            #print(mriImage)
            if (self.val.get() == 1):
                self.listOfWinFrame = 0
                self.listOfWinFrame = list()
                self.listOfWinFrame.append(self.FirstFrame)

                self.listOfWinFrame[0].setCallObject(self.DT)

                res = predictTumor(mriImage)
                
                if res > 0.5:
                    resLabel = tkinter.Label(self.FirstFrame.getFrames(), text="Tumor Detected", height=1, width=20)
                    resLabel.configure(background="White", font=("Comic Sans MS", 16, "bold"), fg="red")
                else:
                    resLabel = tkinter.Label(self.FirstFrame.getFrames(), text="No Tumor", height=1, width=20)
                    resLabel.configure(background="White", font=("Comic Sans MS", 16, "bold"), fg="green")

                resLabel.place(x=700, y=450)

            elif (self.val.get() == 2):
                self.listOfWinFrame = 0
                self.listOfWinFrame = list()
                self.listOfWinFrame.append(self.FirstFrame)

                self.listOfWinFrame[0].setCallObject(self.DT)
                self.listOfWinFrame[0].setMethod(self.DT.removeNoise)
                secFrame = Frames(self, MainWindow, self.wWidth, self.wHeight, self.DT.displayTumor, self.DT)

                self.listOfWinFrame.append(secFrame)


                for i in range(len(self.listOfWinFrame)):
                    if (i != 0):
                        self.listOfWinFrame[i].hide()
                self.listOfWinFrame[0].unhide()

                if (len(self.listOfWinFrame) > 1):
                    self.listOfWinFrame[0].btnView['state'] = 'active'

            else:
                print("Not Working")

    mainObj = Gui()
    return render(request, 'users/UserHome.html', {})

def UserTraining(request):
    from .AlgoProcess import modelTraining
    acc,loss = modelTraining.StartTraining()
    return render(request,'users/UserTraining.html' ,{'acc':acc,'loss':loss})




