#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 08:42:18 2019

@author: davidblair
"""

from unidecode import unidecode
import pkg_resources
import string

ICD_PATH = pkg_resources.resource_filename('vlpi', 'data/ICDData/')


class ICDCode:
    def __init__(self, code, associated_string,is_terminal, parent_code=None):
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
        if code in self.setOfUnusableCodes:
            return self.UnusableICDCodes[self.unusableCodeToIndexMap[code.replace('.','')]]
        else:
            return self.UsableICDCodes[self.usableCodeToIndexMap[code.replace('.','')]]
       
    
    def DeleteCode(self,del_code):
        """
        Removes the ICD code and all children (if exist) from data structure.
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
        Returns the chapter heading for any code in the code book.
        """
        
        code = code.replace('.','')
        currentCode = self.ReturnCodeObject(code)
        while currentCode.parent_code is not None:
            currentCode = self.ReturnCodeObject(currentCode.parent_code.code)
        return str(currentCode)
    
    def ReturnSubsumedTerminalCodes(self,parent_code):
        """
        Returns the list of all terminal codes (ie no children, also encoding in medical recored)
        that are subsumed by some parent code of interest.
        """
        all_child_codes = self.ReturnCodeObject(parent_code).child_codes
        terminal_code_list=[]
        for child in all_child_codes:
            if child.is_terminal==True:
                terminal_code_list+=[child.code]
            else:
                terminal_code_list+=self.ReturnSubsumedTerminalCodes(child.code)
        return terminal_code_list
            
        

    def __init__(self,hierarchyFile=None,chapterFile=None):
        """
        Class that manipulates the ICD10 codebook. It stores the codebook as a
        simple tree (stored as a list called ICDCodes).
            
        
        To initialize the class, expects flat two text files:
        
            1) ICD10_Chapters.txt--chapter heading for all the codes. Manually constructed.
            2) ICd10 codes and hierarchy: icd10cm_order_2018.txt, https://www.cms.gov/Medicare/Coding/ICD10/2018-ICD-10-CM-and-GEMs.html

        By default, the package ships with 2018 version of ICD10. You can upgrade to 2019 (or downgrade for that matter)
        by specifying the path to another ICD10 file. ICD9 codebook could be used instead,
        but you would need to construct data files that match the structure of the ICD10 files.
        
        
        """
        if hierarchyFile==None:
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
                parsedLine+=[line[16:77].strip()]
                
                
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
