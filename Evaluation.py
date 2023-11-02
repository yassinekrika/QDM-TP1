from PIL import Image, ImageTk
import numpy as np
import math

#-------------------------------------------------------- PSNR Globale -------------------------------------------------------------------------------------------------

def PsnrG(ImageOrig,ImageD):

    print("PSNR Globale")
    ResultatPSNR = 0
    Somme = 0
    
    L,H=ImageOrig.size
    MatOrig=np.array(ImageOrig)
    MatD=np.array(ImageD)
    
    for i in range(H):        
        for j in range(L):
            
            Somme = Somme +((int(MatOrig[i,j]) - int(MatD[i,j]))**2)

  
    Max2 = 255**2       
    MSE = (Somme/float(L*H)) 
    M = Max2/float(MSE)     
    ResultatPSNR= 10 * math.log10(M)
    return ResultatPSNR


#-------------------------------------------------------- PSNR Locale --------------------------------------------------------------------------------------------------

def PsnrL(ImageOrig,ImageD,F):

    print ("PSNR Locale")
    L,H=ImageOrig.size
    MatOrig=np.array(ImageOrig)
    MatD=np.array(ImageD)
    nbrCase = L*H/(F*F)
    nbrC = 0
    ResultatPSNRT = 0
    for i in range(0,L//F):
        for j in range(0,H//F):
            Somme = 0
            if ( nbrC < nbrCase ):
                for k in range(0,F):
                    for l in range(0,F):
                        Somme = Somme +((int(MatOrig[j*F+l,i*F+k]) - int(MatD[j*F+l,i*F+k]))**2)
                   
                Max2 = 255**2
                MSE = (Somme/float(F*F))
                M = Max2/float(MSE)
                ResultatPSNRCase= 10 * math.log10(M)
                ResultatPSNRT = ResultatPSNRT + ResultatPSNRCase
                nbrC = nbrC + 1
            else:
                
                break  
                       
    return ResultatPSNRT/nbrCase


#-------------------------------------------------------- SSIM Globale --------------------------------------------------------------------------------------------------

def SsimG(ImageOrig,ImageD):

    print ("SSIM Globale")
    
    ResultatSSIM = 0
    SommeLX = 0
    SommeLY = 0
    SommeCX = 0
    SommeCY = 0

    Somme = 0
    
    L,H=ImageOrig.size
    MatOrig=np.array(ImageOrig)
    MatD=np.array(ImageD)

    
    for i in range(H):        
        for j in range(L):
            
            SommeLX = SommeLX +(int(MatOrig[i,j]))
            SommeLY = SommeLY +(int(MatD[i,j]))

    LuminanceX = (SommeLX/float(L*H))
    LuminanceY = (SommeLY/float(L*H))
    print ("LuminanceX", LuminanceX)
    print ("LuminanceY", LuminanceY)
    
    for i in range(H):        
        for j in range(L):
            
            SommeCX = SommeCX +((int(MatOrig[i,j])- LuminanceX)**2)
            SommeCY = SommeCY +((int(MatD[i,j])- LuminanceY)**2)

    ContrasteX = (SommeCX/(float(L*H)-1))**0.5
    ContrasteY = (SommeCY/(float(L*H)-1))**0.5
    print ("ContrasteX", ContrasteX)
    print ("ContrasteY", ContrasteY)
    
    for i in range(H):        
        for j in range(L):
            
            Somme = Somme +((int(MatOrig[i,j])- LuminanceX)*(int(MatD[i,j])- LuminanceY))

    SimilariteXY = (Somme/(float(L*H)-1))
    print ("SimilariteXY", SimilariteXY)

    
    Lxy = ((2*LuminanceX*LuminanceY)+(0.03*255)**2)/float((LuminanceX)**2+(LuminanceY)**2+(0.03*255)**2)
    print ("Lxy", Lxy)
    Cxy = ((2*ContrasteX*ContrasteY)+(0.03*255)**2)/float((ContrasteX)**2+(ContrasteY)**2+(0.03*255)**2)
    print ("Cxy", Cxy)
    Sxy = ((2*SimilariteXY)+(0.03*255)**2)/float((2*ContrasteX*ContrasteY)+(0.03*255)**2)
    print ("Sxy", Sxy )                        
    ResultatSSIM = Lxy * Cxy * Sxy
    print ("ResultatSSIM", ResultatSSIM)
    
    return ResultatSSIM


#-------------------------------------------------------- SSIM Locale --------------------------------------------------------------------------------------------------

def SsimL(ImageOrig,ImageD,F):

    print("SSIM Locale")
    L,H=ImageOrig.size
    MatOrig=np.array(ImageOrig)
    MatD=np.array(ImageD)
    nbrCase = L*H/(F*F)
    nbrC = 0
    
    ResultatSSIM = 0
    


    
    for i in range(0,L//F):
        for j in range(0,H//F):
            
            SommeLX = 0
            LuminanceX = 0    
            SommeLY = 0
            LuminanceY = 0    
            SommeCX = 0
            ContrasteX = 0
            SommeCY = 0
            ContrasteY = 0
            Somme = 0
            SimilariteXY = 0

            if ( nbrC < nbrCase ):
                for k in range(0,F):
                    for l in range(0,F):
                        SommeLX = SommeLX +int(MatOrig[j*F+l,i*F+k])
                        SommeLY = SommeLY +int(MatD[j*F+l,i*F+k])
                        
                LuminanceX = LuminanceX + ((SommeLX)/(float(F*F))) 
                LuminanceY = LuminanceY + ((SommeLY)/(float(F*F)))
                print ("LuminanceX", LuminanceX)
                print ("LuminanceY", LuminanceY)

                
                for k in range(0,F):
                    for l in range(0,F):
                        SommeCX = SommeCX +(int(MatOrig[j*F+l,i*F+k])- LuminanceX )**2
                        SommeCY = SommeCY +(int(MatD[j*F+l,i*F+k]) - LuminanceY )**2
                        
                ContrasteX = ContrasteX + ((SommeCX)/(float(F*F)-1))**0.5 
                ContrasteY = ContrasteY + ((SommeCY)/(float(F*F)-1))**0.5
                print ("ContrasteX", ContrasteX)
                print ("ContrasteY", ContrasteY)

                
                for k in range(0,F):
                    for l in range(0,F):
                        Somme = Somme +((int(MatOrig[j*F+l,i*F+k])- LuminanceX )*(int(MatD[j*F+l,i*F+k]) - LuminanceY ))
                     
                        
                SimilariteXY = SimilariteXY + ((Somme)/(float(F*F)-1))
                print ("SimilariteXY", SimilariteXY)
                
                Lxy = ((2*LuminanceX*LuminanceY)+(0.03*255)**2)/float((LuminanceX)**2+(LuminanceY)**2+(0.03*255)**2)
                print ("Lxy", Lxy)
                Cxy = ((2*ContrasteX*ContrasteY)+(0.03*255)**2)/float((ContrasteX)**2+(ContrasteY)**2+(0.03*255)**2)
                print ("Cxy", Cxy)
                Sxy = ((2*SimilariteXY)+(0.03*255)**2)/float((2*ContrasteX*ContrasteY)+(0.03*255)**2)
                print ("Sxy", Sxy)                         
                ResultatSSIM = ResultatSSIM + ( Lxy * Cxy * Sxy )
                print ("ResultatSSIM", ResultatSSIM)
     

            else:
                
                break
    
                       
    return ResultatSSIM/nbrCase




