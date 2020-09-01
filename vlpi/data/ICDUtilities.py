#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 08:42:18 2019

@author: davidblair
"""

from unidecode import unidecode
import pkg_resources
import string
import pickle
import pandas as pd

ICD_PATH = pkg_resources.resource_filename('vlpi', 'data/ICDData/')


class ICDCode:
    def __init__(self, code, associated_string,is_terminal, parent_code=None):
        """


        Parameters
        ----------
        code : str
            ICD10 code string.
        associated_string : str
            String defining code in codebook.
        is_terminal : bool
            Indicates whether code is terminal (no children).
        parent_code : bool, optional
            Indicates if code is parent. The default is None.

        Raises
        ------
        ValueError
            If unable to add child codes to known parent code.

        Returns
        -------
        None.

        """


        self.code = code
        self.associated_string = associated_string
        self.parent_code = parent_code
        self.child_codes = []
        self.is_terminal=is_terminal
        self.parent_code = parent_code
        if parent_code is not None:
            if parent_code.is_terminal==False:
                parent_code.child_codes.append(self)
            else:
                raise ValueError("Attempting to add children to terminal node: child = {}, parent = {}".format(code,parent_code.code))

    def __str__(self):
        return self.associated_string

    def __repr__(self):
        return self.code


class ICDUtilities:

    def _convertToUnicode(self,byteString):
        return unidecode(str(byteString,"ISO-8859-1"))

    def _lettersToInt(self,letter,base):
        list(string.ascii_uppercase).index(letter)*base

    def _convertCodeToIntVal(self,code):
        intVal=0
        for base,letter in enumerate(code[::-1]):
            try:
                intVal+=int(letter)*10**(base)
            except ValueError:
                intVal+=int(list(string.ascii_uppercase).index(letter))*10**(base)
        return intVal


    def _findParentInList(self,code,parentList):

        while len(parentList) > 0:
            if parentList[-1] in code:
                return parentList
            else:
                parentList.pop()
        return parentList

    def ReturnCodeObject(self,code):
        """
        Returns full code object (not just string) for a given code.

        Parameters
        ----------
        code : str
            ICD10 code string.

        Returns
        -------
        ICDCode
            ICD10 code class for input string.

        """

        if code in self.setOfUnusableCodes:
            return self.UnusableICDCodes[self.unusableCodeToIndexMap[code.replace('.','')]]
        else:
            return self.UsableICDCodes[self.usableCodeToIndexMap[code.replace('.','')]]


    def DeleteCode(self,del_code):
        """
        Removes the ICD code and all children (if exist) from data structure.

        Parameters
        ----------
        del_code : str
            ICD10 code to delete


        Returns
        -------
        None
        """


        all_del_codes=self._deleteCode(del_code)


        marker_list_usable= [x.code for x in self.UsableICDCodes]
        marker_list_unusable = [x.code for x in self.UnusableICDCodes]


        for del_code in all_del_codes:
            if del_code in self.setOfUnusableCodes:
                self.UnusableICDCodes.pop(marker_list_unusable.index(del_code))
                marker_list_unusable.remove(del_code)
            else:
                self.UsableICDCodes.pop(marker_list_usable.index(del_code))
                marker_list_usable.remove(del_code)

        self.usableCodeToIndexMap=dict(zip(marker_list_usable,range(len(marker_list_usable))))
        self.unusableCodeToIndexMap=dict(zip(marker_list_unusable,range(len(marker_list_unusable))))
        self.setOfUnusableCodes=set(self.unusableCodeToIndexMap.keys())
        self.setOfUsableCodes=set(self.usableCodeToIndexMap.keys())

    def _deleteCode(self,del_code):

        del_code_obj = self.ReturnCodeObject(del_code)
        parent_code = del_code_obj.parent_code
        del_code_list=[del_code]

        if parent_code is not None:
            parent_code.child_codes.remove(del_code_obj)
        if del_code_obj.is_terminal==False:
            for child_code in del_code_obj.child_codes:
                del_code_list+=self._deleteCode(child_code.code)
        return del_code_list



    def AssignCodeToChapter(self,code):
        """
        Returns the chapter heading for input code

        Parameters
        ----------
        code : str
            ICD10 code.

        Returns
        -------
        str
            ICD10 chapter.

        """


        code = code.replace('.','')
        currentCode = self.ReturnCodeObject(code)
        while currentCode.parent_code is not None:
            currentCode = self.ReturnCodeObject(currentCode.parent_code.code)
        return str(currentCode)

    def ReturnSubsumedTerminalCodes(self,parent_code):
        """
        Returns all ICD10 codes that are children of the input code.

        Parameters
        ----------
        parent_code : str
            ICD10 string for parent code. Do not include periods (ie J101 not J10.1)

        Returns
        -------
        terminal_code_list : list
            List of ICD10 codes that are children to parent code.

        """

        all_child_codes = self.ReturnCodeObject(parent_code).child_codes
        terminal_code_list=[]
        for child in all_child_codes:
            if child.is_terminal==True:
                terminal_code_list+=[child.code]
            else:
                terminal_code_list+=self.ReturnSubsumedTerminalCodes(child.code)
        return terminal_code_list



    def __init__(self,useICD10UKBB=False,hierarchyFile=None,chapterFile=None):
        """
        Class that manipulates the ICD10 codebook. It stores the codebook as a simple tree (stored as a list called ICDCodes).


        To initialize the class, expects flat two text files:

            1) ICD10_Chapters.txt--chapter heading for all the codes. Manually constructed.
            2) ICD10 codes and hierarchy: icd10cm_order_2018.txt, https://www.cms.gov/Medicare/Coding/ICD10/2018-ICD-10-CM-and-GEMs.html

        By default, the package ships with 2018 version of ICD10-CM and 2020 version of ICD10 from UK Biobank. You can upgrade to 2019 (or downgrade for that matter) by specifying the path to another ICD10 file. ICD9 codebook could be used instead, but you would need to construct data files that match the structure of the ICD10 files.


        Parameters
        ----------
        useICD10UKBB : bool, optional
            Specifies class to use the UK Biobank version of ICD10 (not ICD10-CM). The default is False.
        hierarchyFile : str, optional
            File path to an alternative code hierarchy file. This may (unlikely) work with other encodings but has not been tested. The default is None.
        chapterFile : str, optional
            File path to an alternative code chapter file (ie main groups of codes). Again, this may work with other encodings but has not been tested. The default is None.

        Raises
        ------
        ValueError
            ValueError raised if unable to parse some line. Prints out the line of interest.

        Returns
        -------
        None.

        """

        if hierarchyFile==None:
            if useICD10UKBB:
                hierarchyFile=ICD_PATH+'icd10_ukbb.txt'
            else:
                hierarchyFile=ICD_PATH+'icd10cm_order_2018.txt'
        if chapterFile==None:
            chapterFile = ICD_PATH+'ICD10_Chapters.txt'

        #quick reference, to avoid having to search full tree for codes.


        #Full list of linked codes
        self.UsableICDCodes=[]
        self.usableCodeToIndexMap={}
        self.setOfUsableCodes=set()

        self.UnusableICDCodes=[]
        self.unusableCodeToIndexMap={}
        self.setOfUnusableCodes=set()
        #first load the chapters
        chapter_breakpoints=[]
        chapter_list=[]
        currentUsableCodeCount = 0
        currentUnusableCodeCount = 0
        with open(chapterFile,'rb') as f:
            f.readline()
            for line in f:
                line=self._convertToUnicode(line)
                line=line.strip('\n').split('\t')
                self.UnusableICDCodes+=[ICDCode('Chapter_'+line[0],line[2],False)]
                start,stop = line[1].split('-')
                chapter_breakpoints+=[self._convertCodeToIntVal(stop[0:3])]
                chapter_list+=['Chapter_'+line[0]]
                self.unusableCodeToIndexMap['Chapter_'+line[0]]=currentUnusableCodeCount
                self.setOfUnusableCodes.add('Chapter_'+line[0])
                currentUnusableCodeCount+=1
        #now load hierarchy file

        with open(hierarchyFile,'rb') as f:
            currentParentList = []
            for line in f:
                line=self._convertToUnicode(line)
                parsedLine=[]
                parsedLine+=[line[0:6].strip()]
                parsedLine+=[line[6:14].strip()]
                parsedLine+=[line[14:16].strip()]
                parsedLine+=[line[77:].strip()]


                currentParentList = self._findParentInList(parsedLine[1],currentParentList)

                if len(currentParentList) == 0:
                    intVal = self._convertCodeToIntVal(parsedLine[1][0:3])
                    try:
                        icd_chapter = chapter_list[next(x[0] for x in enumerate(chapter_breakpoints) if intVal <= x[1])]
                    except StopIteration:
                        raise ValueError('{}'.format(parsedLine[1]))
                    currentParentList +=[icd_chapter]

                if int(parsedLine[2])==1:
                    self.UsableICDCodes+=[ICDCode(parsedLine[1],parsedLine[3],True,self.ReturnCodeObject(currentParentList[-1]))]
                    self.usableCodeToIndexMap[parsedLine[1]]=currentUsableCodeCount
                    self.setOfUsableCodes.add(parsedLine[1])
                    currentUsableCodeCount+=1
                else:
                    self.UnusableICDCodes+=[ICDCode(parsedLine[1],parsedLine[3],False,self.ReturnCodeObject(currentParentList[-1]))]
                    self.unusableCodeToIndexMap[parsedLine[1]]=currentUnusableCodeCount
                    self.setOfUnusableCodes.add(parsedLine[1])
                    currentUnusableCodeCount+=1
                    currentParentList+=[parsedLine[1]]


class ICD10TranslationMap:

    def _buildTranslationTable(self):

        translation_table={'Primary Code':[],'Secondary Code(s)':[],'Relationship':[]}
        for code in self.primaryEncoding.UsableICDCodes:
            #first check if there is 1:1 mapping between codes
            if code.code in self.secondaryEncoding.setOfUsableCodes:
                translation_table['Primary Code']+=[code.code]
                translation_table['Secondary Code(s)']+=[set([code.code])]
                translation_table['Relationship']+=['Direct']
            else:
                parent = code.parent_code
                if parent.code in self.secondaryEncoding.setOfUsableCodes:
                    translation_table['Primary Code']+=[code.code]
                    translation_table['Secondary Code(s)']+=[set([parent.code])]
                    translation_table['Relationship']+=['Parent']
                else:
                    if len(code.child_codes)>0:
                        child_code_names = [x.code for x in code.child_codes]
                        allowed_child_codes = set(child_code_names).intersection(self.secondaryEncoding.setOfUsableCodes)
                        if len(allowed_child_codes)>0:
                            translation_table['Primary Code']+=[code.code]
                            translation_table['Secondary Code(s)']+=[allowed_child_codes]
                            translation_table['Relationship']+=['Child']

        translation_table=pd.DataFrame(translation_table)
        translation_table.set_index('Primary Code',drop=False,inplace=True)
        return translation_table



    def __init__(self,primaryEncoding=None,secondaryEncoding=None):
        """
        Builds translation map between two ICD Utilities with at least some shared codes by taking advantage of shared hierarchical structure.

        If primaryEncoding and secondaryEncoding are unspecified, class creates a map between ICD10-CM and ICD10 (UKBB)

        Parameters
        ----------
        primaryEncoding : ICDUtilities, optional
            First encoding. The default is None.
        secondaryEncoding : ICDUtilities, optional
            Second encoding. The default is None.

        Returns
        -------
        None.

        """

        if (primaryEncoding is not None) or (secondaryEncoding is not None):
            assert (secondaryEncoding is not None) and (secondaryEncoding is not None), "Must specify primary and secondary encoding if providing one or the other."
            self.primaryEncoding=primaryEncoding
            self.secondaryEncoding=secondaryEncoding

        if primaryEncoding is None:
            try:
                with open(ICD_PATH+'icd10cm_to_ukbb.pth','rb') as f:
                    self.EncodingCoversionTable = pickle.load(f)
            except FileNotFoundError:
                self.primaryEncoding = ICDUtilities()
                self.secondaryEncoding=ICDUtilities(useICD10UKBB=True)

                self.EncodingCoversionTable=self._buildTranslationTable()
                self.EncodingCoversionTable.to_pickle(ICD_PATH+'icd10cm_to_ukbb.pth')
        else:
            self.EncodingCoversionTable=self._buildTranslationTable()



    def ReturnConversionSet(self,primaryCode,includeRelationship=False):
        """
        Returns set of codes that represent the mapping of the primary code to the new encoding system.

        Parameters
        ----------
        primaryCode : str
            Diagnostic code to be converted .
        includeRelationship : bool, optional.
            Specicies whether to return the relationship type in addition to code. The default is False.

        Returns
        -------
        set
            Set of codes aligned to the code of interest.

        """

        if includeRelationship:
            look_up=['Secondary Code(s)','Relationship']
        else:
            look_up='Secondary Code(s)'
        try:
            return self.EncodingCoversionTable.loc[primaryCode][look_up]
        except KeyError:
            return set([])
