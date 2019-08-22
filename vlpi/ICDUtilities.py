#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 08:42:18 2019

@author: davidblair
"""

from unidecode import unidecode

class ICDUtilities:
    def _convertToUnicode(self,byteString):
        return unidecode(str(byteString,"ISO-8859-1"))
        
    
    def __init__(self,ICD9File,ICD10File):
        """
        To initialize the class, expects flat text files as downloaded from CMS website.
        
        ICD9: https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes/codes.html
        ICD10: https://www.cms.gov/Medicare/Coding/ICD10/2018-ICD-10-CM-and-GEMs.html
    
        """
        self.ICD9Dict={}
        self.ICD10Dict={}
        self.ICD9_to_IntDict={}
        self.ICD10_to_IntDict={}
        self.Int_to_ICD10Dict={}
        self.Int_to_ICD9Dict={}
        
        
        icd9Text=open(ICD9File,'rb')
        icd10Text=open(ICD10File,'rb')
        
        currentInt=0
        for line in icd9Text:
            line=self._convertToUnicode(line)
            line=line.strip('\n')
            self.ICD9Dict[line[0:6].strip()]=line[6:].strip()
            self.ICD9_to_IntDict[line[0:6].strip()]=currentInt
            self.Int_to_ICD9Dict[currentInt]=line[0:6].strip()
            currentInt+=1
        icd9Text.close()
        
        currentInt=0
        for line in icd10Text:
            line=self._convertToUnicode(line)
            line=line.strip('\n')
            self.ICD10Dict[line[0:7].strip()]=line[7:].strip()
            self.ICD10_to_IntDict[line[0:7].strip()]=currentInt
            self.Int_to_ICD10Dict[currentInt]=line[0:7].strip()
            currentInt+=1
        icd9Text.close()
        
    def returnICD9String(self, code):
        if '.' in code:
            code.replace('.','')
        try:
            return self.ICD9Dict[code]
        except KeyError:
            raise ValueError("No matching ICD9 code: "+code)
    
    def returnICD10String(self,code):
        if '.' in code:
            code.replace('.','')
        try:
            return self.ICD10Dict[code]
        except KeyError:
            raise ValueError("No matching ICD10 code: "+code)
            
    def returnICD9Int(self, code):
        if '.' in code:
            code.replace('.','')
        try:
            return self.ICD9_to_IntDict[code]
        except KeyError:
            raise ValueError("No matching ICD9 code: "+code)
    
    def returnICD10Int(self,code):
        if '.' in code:
            code.replace('.','')
        try:
            return self.ICD10_to_IntDict[code]
        except KeyError:
            raise ValueError("No matching ICD10 code: "+code)
    
    



    
