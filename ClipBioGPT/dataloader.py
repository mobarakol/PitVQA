import os
import glob

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from transformers import CLIPProcessor, BioGptTokenizer


class PitVis24VQA(Dataset):
    def __init__(self, seq, folder_head, folder_tail):
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.tokenizer.pad_token = self.tokenizer.eos_token

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
        qa_full_path = Path(self.vqas[idx][0])
        seq_path = qa_full_path.parents[2]
        video_num = qa_full_path.parts[-2]
        file_name = self.vqas[idx][0].split('/')[-1]

        # img
        img_loc = os.path.join(seq_path, 'images', video_num, file_name.split('.')[0] + '.png')
        img = self.clip_processor(images=Image.open(img_loc), return_tensors="pt")

        # question and answer
        question_text = self.vqas[idx][1].split('|')[0]
        question = self.tokenizer(text=question_text, return_tensors="pt", padding='max_length', max_length=25)
        answer_text = self.vqas[idx][1].split('|')[1]
        label = self.labels.index(str(answer_text))

        return img_loc, img, question, label


class EndoVis18VQAGPTClassification(Dataset):
    def __init__(self, seq, folder_head, folder_tail):
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.tokenizer.pad_token = self.tokenizer.eos_token

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
        qa_full_path = Path(self.vqas[idx][0])
        seq_path = qa_full_path.parents[2]
        file_name = self.vqas[idx][0].split('/')[-1]

        # img
        img_loc = os.path.join(seq_path, 'left_fr', file_name.split('_')[0] + '.png')
        img = self.clip_processor(images=Image.open(img_loc), return_tensors="pt")

        # question and answer
        question_text = self.vqas[idx][1].split('|')[0]
        answer_text = self.vqas[idx][1].split('|')[1]
        question = self.tokenizer(text=question_text, return_tensors="pt", padding='max_length', max_length=25)
        label = self.labels.index(str(answer_text))

        return img_loc, img, question, label
