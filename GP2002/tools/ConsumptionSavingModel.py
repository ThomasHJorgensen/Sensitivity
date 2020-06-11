import os
import ctypes as ct
import pickle
import numpy as np
from numba import int32, double, boolean

def convert_to_dict(someclass,somelist):

    keys = [var[0] for var in somelist]
    values = [getattr(someclass,key) for key in keys]
    return {key:val for key,val in zip(keys,values)}

class ConsumptionSavingModel():
    
    def save(self): # save the model parameters and solution

        # a. save parameters pickle
        par_dict = convert_to_dict(self.par,self.parlist)
        with open(f'data/{self.name}_{self.solmethod}.p', 'wb') as f:
            pickle.dump(par_dict, f)

        # b. solution
        sol_dict = convert_to_dict(self.sol,self.sollist)
        np.savez(f'data/{self.name}_{self.solmethod}.npz', **sol_dict)
    
    def load(self): # load the model parameters and solution

        # a. parameters
        with open(f'data/{self.name}_{self.solmethod}.p', 'rb') as f:
            self.par_dict = pickle.load(f)
        for key,val in self.par_dict.items():
            setattr(self.par,key,val)

        # b. solution
        with np.load(f'data/{self.name}_{self.solmethod}.npz') as data:
            for key in data.files:
                setattr(self,key,data[key])

    def __str__(self): # called when model is printed

        # a. keys and values in ParClass
        keys = [var[0] for var in self.parlist]
        values = [getattr(self.par,key) for key in keys]

        # b. create description
        description = ''
        for var,val in zip(self.parlist,values):
            if var[1] == int32 or var[1] == double:
                description += f'{var[0]} = {val}\n'
            elif var[1] == double[:]:
                description += f'{var[0]} = [array of doubles]\n'

        return description 

    def setup_cpp(self,compiler='msvs_2017'):
        """ setup interface to cpp files """

        # a. compiler
        self.compiler = compiler
        
        # b. dictionary of cppfiles
        self.cpp_files = dict()

        # c. ctypes version of par and sol classes
        parlist_fields, parlist_txt = self.get_fields(self.parlist)
        class parcpp(ct.Structure):
            _fields_ = parlist_fields
        
        sollist_fields, sollist_txt = self.get_fields(self.sollist)
        class solcpp(ct.Structure):
            _fields_ = sollist_fields
    
        self.parcpp = parcpp
        self.solcpp = solcpp

        # d. write structs in cpp (ordering of fields MUST be the same)
        with open('cppfuncs\\mystructs.cpp', 'w') as cppfile:

            cppfile.write('typedef struct par_struct\n') 
            cppfile.write('{\n')
            cppfile.write(parlist_txt)
            cppfile.write('} par_struct;\n\n')
            
            cppfile.write('typedef struct sol_struct\n') 
            cppfile.write('{\n')
            cppfile.write(sollist_txt)
            cppfile.write('} sol_struct;\n')

    def compile_cpp(self,filename):
        """ write and execute compile.bat compiling C++ file in filename """

        # a. delink if necessary
        if filename in self.cpp_files:
            if self.cpp_files[filename] != None:
                self.delink_cpp()

        # b. strings
        pwd_str = 'cd "' + os.getcwd() + '"'
        if self.compiler == 'msvs_2017':
            path_str = 'cd "C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/"'
            version_str = 'call vcvarsall.bat x64'
            nlopt_lib = '' #'cppfuncs/nlopt-2.4.2-dll64/libnlopt-0.lib'
            compile_str = f'cl {nlopt_lib} /LD /EHsc /Ox /openmp {filename}'
        else:
            raise ValueError('unknown compiler')

        # c. write bat file
        with open('compile.bat', 'w') as txtfile:
            for now_str in [path_str,version_str,pwd_str,compile_str]:
                print('{}'.format(now_str), file=txtfile)

        # d. compile
        result = os.system('compile.bat')
        if result == 0:
            print('cpp files compiled successfully')
        else: 
            raise ValueError('cpp files can not be compiled')

        # e. clean up
        os.remove('compile.bat')
        filename_raw = os.path.splitext(os.path.basename(filename))[0]
        if self.compiler == 'msvs_2017':
            os.remove(f'{filename_raw}.obj')
            os.remove(f'{filename_raw}.lib')
            os.remove(f'{filename_raw}.exp')

    def get_fields(self,nblist):
        """ construct ctypes list of fields from list of fields for numba class """
        
        ctlist = []
        cttxt = ''
        for nbelem in nblist:
            if nbelem[1] == int32:
                ctlist.append((nbelem[0],ct.c_long))
                cttxt += f' int {nbelem[0]};\n'
            elif nbelem[1] == double:
                ctlist.append((nbelem[0],ct.c_double))          
                cttxt += f' double {nbelem[0]};\n'
            elif nbelem[1] == boolean:
                ctlist.append((nbelem[0],ct.c_bool))
            elif nbelem[1].dtype == int32:
                ctlist.append((nbelem[0],ct.POINTER(ct.c_long)))               
                cttxt += f' int *{nbelem[0]};\n'
            elif nbelem[1].dtype == double:
                ctlist.append((nbelem[0],ct.POINTER(ct.c_double)))
                cttxt += f' double *{nbelem[0]};\n'
            else:
                raise ValueError(f'unknown type for {nbelem[0]}')
        
        return ctlist,cttxt
    
    def get_pointers(self,nbclass,ctclass):
        
        """ construct ctypes class with pointers from numba class """
        
        p_ctclass = ctclass()
        
        for field in ctclass._fields_:
            
            key = field[0]                
            val = getattr(nbclass,key)
            if isinstance(field[1](),ct.c_long):
                setattr(p_ctclass,key,val)
            elif isinstance(field[1](),ct.POINTER(ct.c_long)):
                assert np.issubdtype(val.dtype, np.int32)            
                setattr(p_ctclass,key,np.ctypeslib.as_ctypes(val.ravel()))
            elif isinstance(field[1](),ct.c_double):
                setattr(p_ctclass,key,val)            
            elif isinstance(field[1](),ct.POINTER(ct.c_double)):
                assert np.issubdtype(val.dtype, np.double)
                setattr(p_ctclass,key,np.ctypeslib.as_ctypes(val.ravel()))
            elif isinstance(field[1](),ct.c_bool):
                setattr(p_ctclass,key,val)
            else:
                raise ValueError(f'no such type, variable {key}')
        
        return p_ctclass

    def link_cpp(self,filename,funcs,do_compile=True):

        # a. compile
        if do_compile:
            self.compile_cpp(filename)

        # b. link
        filename_raw = os.path.splitext(os.path.basename(filename))[0]
        self.cpp_files[filename] = ct.cdll.LoadLibrary(f'{filename_raw}.dll')
        print('cpp files linked successfully')

        # c. set input and output types
        for func in funcs:
            funcnow = getattr(self.cpp_files[filename],func)
            funcnow.restype = None
            funcnow.argtypes = [ct.POINTER(self.parcpp),ct.POINTER(self.solcpp)]
                
    def delink_cpp(self,filename):

        # 1. get handle
        handle = self.cpp_files[filename]._handle

        # 2. delete linking variable
        del self.cpp_files[filename]
        self.cpp_files[filename] = None

        # 3. free handle
        ct.windll.kernel32.FreeLibrary.argtypes = [ct.wintypes.HMODULE]
        ct.windll.kernel32.FreeLibrary(handle)
        print('cpp files delinked successfully')

    def call_cpp(self,filename,func):
            
        p_par = self.get_pointers(self.par,self.parcpp)
        p_sol = self.get_pointers(self.sol,self.solcpp)
        
        funcnow = getattr(self.cpp_files[filename],func)
        funcnow(ct.byref(p_par),ct.byref(p_sol))