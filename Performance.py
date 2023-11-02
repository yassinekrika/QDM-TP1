from PIL import Image, ImageTk
import numpy as np
import math

#-------------------------------------------------------- Coefficient de correlation lineaire -------------------------------------------------------------------------------------------------

def CC(ResSubjective,ResObjective):

    nbrVO = 10
    nbrVD = 15
    SommeMos = 0
    SommeMobj = 0
    SommeRMos = 0
    SommeRMobj = 0
    SommeCC = 0
    Nbjk = (nbrVO * nbrVD)
    
    for i in range(nbrVO):
        for j in range(nbrVD):
    
            SommeMos = SommeMos + float(ResSubjective[(i * nbrVD) + j])
            SommeMobj = SommeMobj + float(ResObjective[(i * nbrVD) + j])
               
    MOS = ((SommeMos)/Nbjk)
    Mobj = ((SommeMobj)/Nbjk)
    
    for i in range(nbrVO):
        for j in range(nbrVD):
            
            SommeRMos = SommeRMos + ( float(ResSubjective[(i * nbrVD) + j]) - float(MOS) )**2
            SommeRMobj = SommeRMobj + ( float(ResObjective[(i * nbrVD) + j]) - float(Mobj))**2
            SommeCC = SommeCC + (( float(ResSubjective[(i * nbrVD) + j]) - float(MOS) )*( float(ResObjective[(i * nbrVD) + j]) - float(Mobj)))
    
    RMOS = SommeRMos/(Nbjk-1)
    RMobj = SommeRMobj/(Nbjk-1)
    CC = SommeCC/(Nbjk*(math.sqrt(RMOS*RMobj)))
    
    return CC
    



#-------------------------------------------------------- Coefficient de correlation de rang --------------------------------------------------------------------------------------------------

def CCR(ResSubjective,ResObjective):

    nbrVO = 10
    nbrVD = 15
    SommeCCR = 0
    Nbjk = (nbrVO * nbrVD)
    
    for i in range(nbrVO):
        for j in range(nbrVD):
    
            SommeCCR = SommeCCR + (float(ResSubjective[(i * nbrVD) + j]) - float(ResObjective[(i * nbrVD) + j]))**2

    CCR = 1 - ((6 * SommeCCR )/(Nbjk**3-Nbjk))
    
    return CCR
                       



#-------------------------------------------------------- erreur de prediction de la qualite --------------------------------------------------------------------------------------------------

def Erreur(ResSubjective,ResObjective):

    nbrVO = 10
    nbrVD = 15
    SommeErreur = 0
    Nbjk = (nbrVO * nbrVD)

    for i in range(nbrVO):
        for j in range(nbrVD):
    
            SommeErreur = SommeErreur + (float(ResSubjective[(i * nbrVD) + j]) - float(ResObjective[(i * nbrVD) + j]))**2
    
    Erreur = math.sqrt((SommeErreur/(Nbjk-1)))
    
    return Erreur

#-------------------------------------------------------- erreur de prediction de la qualite ponderee par intervalle de confiance a 95% --------------------------------------------------------------------------------------------------

def ErreurC(ResSubjective,ResObjective):

    nbrVO = 10
    nbrVD = 15
    SommeErreurC = 0
    Nbjk = (nbrVO * nbrVD)
    Icjk = 95

    for i in range(nbrVO):
        for j in range(nbrVD):
    
            SommeErreurC = SommeErreurC + ((float(ResSubjective[(i * nbrVD) + j]) - float(ResObjective[(i * nbrVD) + j]))/(Icjk + 0.025))**2
    
    ErreurC = math.sqrt((SommeErreurC/(Nbjk-1)))
    
    return ErreurC

#-------------------------------------------------------- Coefficient de Kurtosis --------------------------------------------------------------------------------------------------

def CKurtosis(ResSubjective,ResObjective):

    nbrVO = 10
    nbrVD = 15
    SommeCKurtosisA = 0
    SommeCKurtosisB = 0
    Nbjk = (nbrVO * nbrVD)


    for i in range(nbrVO):
        for j in range(nbrVD):
    
            SommeCKurtosisA = SommeCKurtosisA + (float(ResSubjective[(i * nbrVD) + j]) - float(ResObjective[(i * nbrVD) + j]))**4
            SommeCKurtosisB = SommeCKurtosisB + (float(ResSubjective[(i * nbrVD) + j]) - float(ResObjective[(i * nbrVD) + j]))**2
    
    CKurtosis = ((SommeCKurtosisA)/(Nbjk))/((SommeCKurtosisB)/(Nbjk))**2
    
    return CKurtosis


