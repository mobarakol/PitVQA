import os
import glob

import h5py
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision.transforms.functional import InterpolationMode


class PitVis24VQA(Dataset):
    def __init__(self, seq, folder_head, folder_tail, patch_size=4):
        self.patch_size = patch_size

        # files, question and answers
        filenames = []
        for curr_seq in seq:
            filenames = filenames + glob.glob(folder_head + curr_seq + folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines:
                answer = line.split('|')[1]
                if answer not in ['no_visible_instrument', 'no_secondary_instrument']:
                    self.vqas.append([file, line])
        print('Total files: %d | Total question: %.d' % (len(filenames), len(self.vqas)))

        # Labels
        self.labels = [
            'nasal_corridor_creation', 'anterior_sphenoidotomy', 'septum_displacement', 'sphenoid_sinus_clearance',
            'sellotomy', 'haemostasis', 'synthetic_graft_placement', 'durotomy', 'tumour_excision',
            'fat_graft_placement', 'gasket_seal_construct', 'dural_sealant', 'nasal_packing', 'debris_clearance',
            'end_of_step',  # 15 steps
            'nasal_sphenoid', 'sellar', 'closure',  'end_of_phase',  # 4 phases
            'suction', 'freer_elevator', 'pituitary_rongeurs', 'spatula_dissector', 'kerrisons', 'cottle',
            'haemostatic_foam', 'micro_doppler', 'nasal_cutting_forceps', 'stealth_pointer', 'irrigation_syringe',
            'retractable_knife', 'dural_scissors', 'ring_curette', 'cup_forceps', 'bipolar_forceps', 'tissue_glue',
            'surgical_drill',  # 18 instruments
            'zero', 'one', 'two',  # 3 number of instruments
            'top-left', 'top-right', 'centre', 'bottom-left', 'bottom-right',  # 5 positions
            'The middle and superior turbinates are laterally displaced',
            'The sphenoid ostium is identified and opened', 'The septum is displaced until the opposite ostium is seen',
            'The sphenoid sinus is opened, with removal of sphenoid septations to expose the face of the sella and mucosa',
            'Haemostasis is achieved with a surgiflo, a bipolar cautery, and a spongostan placement',
            'The sella is identified, confirmed and carefully opened', 'A cruciate durotomy is performed',
            'The tumour is seen and removed in a piecemeal fashion', 'spongostan, tachosil and duragen placement',
            'A fat graft is placed over the defact', 'Evicel and Adherus dural sealant are applied',
            'Debris is cleared from the nasal cavity and choana', 'A MedPor implant and a fascia lata graft are placed',
            'The nasal cavity is packed with Bismuth soaked ribbon gauze'  # 14 operations
        ]

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        # process path
        img_loc = 'dummy_loc'
        loc = self.vqas[idx][0].split('/')
        visual_feature_loc = os.path.join('/', loc[1], loc[2], loc[3], loc[4], loc[5], 'visual_features',
                                          (str(self.patch_size) + 'x' + str(self.patch_size)), loc[7],
                                          loc[-1].split('.')[0] + '.hdf5')

        # visual features
        frame_data = h5py.File(visual_feature_loc, 'r')
        visual_features = torch.from_numpy(frame_data['visual_features'][:])

        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        answer = self.vqas[idx][1].split('|')[1]
        label = self.labels.index(str(answer))

        return img_loc, visual_features, question, label


class EndoVis18VQAGPTClassification(Dataset):
    def __init__(self, seq, folder_head, folder_tail, patch_size=4):
        self.patch_size = patch_size

        # files, question and answers
        filenames = []
        for curr_seq in seq:
            filenames = filenames + glob.glob(folder_head + str(curr_seq) + folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines:
                self.vqas.append([file, line])
        print('Total files: %d | Total question: %.d' % (len(filenames), len(self.vqas)))

        # Labels
        self.labels = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                       'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction',
                       'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                       'left-top', 'right-top', 'left-bottom', 'right-bottom']

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        # process path
        img_loc = 'dummy_loc'
        loc = self.vqas[idx][0].split('/')
        visual_feature_loc = os.path.join('/', loc[1], loc[2], loc[3], loc[4], loc[5], 'visual_features',
                                          (str(self.patch_size) + 'x' + str(self.patch_size)), loc[6],
                                          loc[-1].split('_')[0] + '.hdf5')

        # visual features
        frame_data = h5py.File(visual_feature_loc, 'r')
        visual_features = torch.from_numpy(frame_data['visual_features'][:])

        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        answer = self.vqas[idx][1].split('|')[1]
        label = self.labels.index(str(answer))

        return img_loc, visual_features, question, label
