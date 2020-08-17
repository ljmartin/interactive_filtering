

#imports:
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import PIL.ImageTk as ImageTk
from PIL import Image
# import ipywidgets as widgets
# from ipywidgets import interact, interact_manual

import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdFingerprintGenerator, PandasTools, Descriptors, FilterCatalog
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import DataStructs
from rdkit.Chem.FilterCatalog import *

import sascorer
from rdkit.Chem.Descriptors import qed

#for clustering (not essential:)
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering

# predicted = pd.read_csv('random_predicted.csv')
# true_pos = pd.read_csv('random_known.csv')

# ##Adding a 'molecules' column using rdkit pandastools:
# PandasTools.AddMoleculeColumnToFrame(predicted,'canonical_smiles','molecules')
# PandasTools.AddMoleculeColumnToFrame(true_pos,'canonical_smiles','molecules')

# #generate fingerprints of predicted ligands and known ligands:
# gen_mo = rdFingerprintGenerator.GetMorganGenerator(fpSize=2048, radius=2)
# predicted_fps = [gen_mo.GetFingerprint(mol) for mol in predicted['molecules']]
# true_fps = [gen_mo.GetFingerprint(mol) for mol in true_pos['molecules']]

#create a list holding the highest similarity to a known ligand.
# similarities= list()
# for count, mol in enumerate(predicted_fps):
#     tanimoto_values = ([DataStructs.TanimotoSimilarity(mol, i) for i in true_fps])
#     index_of_highest = np.argmax(tanimoto_values)
#     similarities.append(tanimoto_values[index_of_highest])
    
#create a list holding the 'synthetic accessibility score'
#reference: https://doi.org/10.1186/1758-2946-1-8
#module code is in: https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score
# sa_score = [sascorer.calculateScore(i) for i in list(predicted['molecules'])]


# #create a list holding the QED drug-likeness score
# #reference: https://doi.org/10.1038/nchem.1243
# qeds = [qed(mol) for mol in predicted['molecules']]



# #create a list holding logp:
# logp = [Descriptors.MolLogP(m) for m in predicted['molecules']]


# #filter catalog usage instructions are here: https://github.com/rdkit/rdkit/pull/536
# params = FilterCatalogParams()
# params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
# catalog = FilterCatalog(params)
# brenk = np.array([catalog.HasMatch(m) for m in predicted['molecules']])


#add these lists as columns to the 'predicted' pd.DataFrame
# predicted['similarities']=similarities
# predicted['sa_score']=sa_score
# predicted['qeds']=qeds
# predicted['logp']=logp


    

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        open_preds_button = Button(window, text = "Open predicted Smiles", command=self.open_predicted)
        open_pos_button = Button(window, text = "Open True positive", command=self.open_true_pos)
        cluster_button = Button(window, text = "cluster", command=self.generate_fingerprints_and_create_list)

        self.similarity_slider = Scale(window, from_=0.1, to=1.0, resolution=0.05, orient=HORIZONTAL, label="Similarity", command=self.get_top_ligands)
        self.LogP_slider = Scale(window, from_=0.1, to=12.0, resolution=0.5, orient=HORIZONTAL, label="LogP", command=self.get_top_ligands)

        self.sa_score_slider = Scale(window, from_=1, to =9.0,  resolution=0.5,  orient=HORIZONTAL, label="SA score", command=self.get_top_ligands)
        self.qed_slider = Scale(window, from_=0.1, to =0.6, resolution=0.05,  orient=HORIZONTAL, label = "QED", command=self.get_top_ligands)
        self.dist_thresh_slider = Scale(window, from_=0.1, to =1.0, resolution=0.05,  orient=HORIZONTAL, label="Distance Thresh", command=self.get_top_ligands)
        self.num_mol_slider = Scale(window, from_=1, to =50,  orient=HORIZONTAL, label = "Num Molecules", command=self.get_top_ligands)
        self.width = 960
        self.height = 540
        myCanvas = Canvas(self.window, bg="white", height=self.height, width=self.width)
        myCanvas.pack(side='right')
        open_preds_button.pack(side='top')
        open_pos_button.pack(side='top')
        cluster_button.pack(side='top')
        self.similarity_slider.pack(side='top')
        self.sa_score_slider.pack(side='top')
        self.qed_slider.pack(side='top')
        self.dist_thresh_slider.pack(side='top')
        self.num_mol_slider.pack(side='top')
        self.myCanvas = myCanvas
        self.window.mainloop()

    def open_true_pos(self):
        true_pos = filedialog.askopenfilename()
        self.true_pos = pd.read_csv(true_pos)
        ##Adding a 'molecules' column using rdkit pandastools:
        PandasTools.AddMoleculeColumnToFrame(self.true_pos,'canonical_smiles','molecules')

    def open_predicted(self):
        predicted = filedialog.askopenfilename()
        self.predicted = pd.read_csv(predicted)
        ##Adding a 'molecules' column using rdkit pandastools:
        PandasTools.AddMoleculeColumnToFrame(self.predicted,'canonical_smiles','molecules')

    def generate_fingerprints_and_create_list(self):
        #generate fingerprints of predicted ligands and known ligands:
        gen_mo = rdFingerprintGenerator.GetMorganGenerator(fpSize=2048, radius=2)
        predicted_fps = [gen_mo.GetFingerprint(mol) for mol in self.predicted['molecules']]
        true_fps = [gen_mo.GetFingerprint(mol) for mol in self.true_pos['molecules']]
        similarities= list()
        for count, mol in enumerate(predicted_fps):
            tanimoto_values = ([DataStructs.TanimotoSimilarity(mol, i) for i in true_fps])
            index_of_highest = np.argmax(tanimoto_values)
            similarities.append(tanimoto_values[index_of_highest])
        #module code is in: https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score
        sa_score = [sascorer.calculateScore(i) for i in list(self.predicted['molecules'])]
        #create a list holding the QED drug-likeness score
        #reference: https://doi.org/10.1038/nchem.1243
        qeds = [qed(mol) for mol in self.predicted['molecules']]
        #create a list holding logp:
        logp = [Descriptors.MolLogP(m) for m in self.predicted['molecules']]
        #filter catalog usage instructions are here: https://github.com/rdkit/rdkit/pull/536
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        catalog = FilterCatalog(params)
        self.brenk = np.array([catalog.HasMatch(m) for m in self.predicted['molecules']])
        #add these lists as columns to the 'predicted' pd.DataFrame
        self.predicted['similarities']=similarities
        self.predicted['sa_score']=sa_score
        self.predicted['qeds']=qeds
        self.predicted['logp']=logp
        print(self.predicted['logp']<6)
        shortlist_mask = ((self.predicted['similarities']<0.2) & 
                        (self.predicted['sa_score']<4) &
                        (self.predicted['qeds'] > 0.25) &
                        (self.predicted['logp']< 6) &
                        (~self.brenk) )
    def get_top_ligands(self, ignore):
        similarity =  self.similarity_slider.get() 
        sa_score = self.sa_score_slider.get()
        qeds = self.qed_slider.get() 
        logp = self.LogP_slider.get()
        # dist_thresh_slider 
        num_mols = self.num_mol_slider.get()
        shortlist_mask = ((self.predicted['similarities']<similarity) & 
                        (self.predicted['sa_score']<sa_score) &
                        (self.predicted['qeds'] > qeds) &
                        (self.predicted['logp']<logp) &
                        (~self.brenk) )

        my_mols =  Draw.MolsToGridImage(self.predicted[shortlist_mask]['molecules'].iloc[:num_mols],
                                    maxMols=num_mols,
                                molsPerRow=6)
        my_mols.thumbnail((self.width, self.height), Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(my_mols)

        self.myCanvas.create_image(self.width/2, self.height/2, image=self.image, anchor=CENTER, tags = 'bg_image')
        self.myCanvas.update()


win = Tk()
App(win, "Lewis's Molecule Machine")

