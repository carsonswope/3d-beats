GENCODES= [
    # atomicAdd(double*, double) only introduced w/ arch 60.
    # need to put in own implementation if need to backport
    # 'arch=compute_52,code=sm_52',
    # 'arch=compute_53,code=sm_53',
    'arch=compute_60,code=sm_60',
    'arch=compute_60,code=compute_60',
    'arch=compute_61,code=sm_61',
    'arch=compute_62,code=sm_62',
    'arch=compute_70,code=sm_70',
    'arch=compute_72,code=sm_72',
    'arch=compute_75,code=sm_75',
    'arch=compute_80,code=sm_80',
    'arch=compute_86,code=sm_86',
    'arch=compute_86,code=compute_86',
]
