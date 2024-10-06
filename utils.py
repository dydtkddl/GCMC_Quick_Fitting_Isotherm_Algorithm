import os
import subprocess
import ast 
import argparse
from math import sin, cos, sqrt
import numpy as np
import os
import django
from asgiref.sync import sync_to_async
import asyncio
import argparse 
from types import SimpleNamespace
import time
params = {
    "simulation_dir": '"${PYRAS}/Quick_Fitting_Algorithm/simulations/example01"',
    "pressure_point": 50000,
    "mof_cif_name": '"[CoreMOF]ABAYIO_clean.cif"',
    "cut_off": 12.5,
    "base_template_path": '"${PYRAS}/Quick_Fitting_Algorithm/base_templates/base_template_gcmc1005.txt"',
    "force_field": '"NayeonFollowup"',
    "nunber_of_cycles": 25000,
    "nunber_of_initial_cycles": 2000,
    "adsorbate": '"O2"',
    "temperature": 293.15,
    "molecule_definition": '"ExampleDefinitions"',
    "translation_probability": 0.5,
    "reinsertion_probability": 0.5,
    "swap_probability": 1.0,
    "raspa_dir": "${RASPA}",
}



# Django 설정 로드
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()
from app.models import GCMCResults,IsothermFittingParameter

def cif2Ucell(cif, cutoff, Display=False):
    pi = np.pi
    deg2rad = pi / 180.0

    with open(cif) as f_tmp:
        f_cont = f_tmp.readlines()

    n_a = len("_cell_length_a")
    n_b = len("_cell_length_b")
    n_c = len("_cell_length_c")
    n_alp = len("_cell_angle_alpha")
    n_bet = len("_cell_angle_beta")
    n_gam = len("_cell_angle_gamma")

    count_compl = 0
    for line in f_cont:
        if len(line) > n_a and line[:n_a] == "_cell_length_a":
            a = float(line.split()[1])
            count_compl += 1
        if len(line) > n_b and line[:n_b] == "_cell_length_b":
            b = float(line.split()[1])
            count_compl += 1
        if len(line) > n_c and line[:n_c] == "_cell_length_c":
            c = float(line.split()[1])
            count_compl += 1
        if len(line) > n_alp and line[:n_alp] == "_cell_angle_alpha":
            alpha = float(line.split()[1]) * deg2rad
            count_compl += 1
        if len(line) > n_bet and line[:n_bet] == "_cell_angle_beta":
            beta = float(line.split()[1]) * deg2rad
            count_compl += 1
        if len(line) > n_gam and line[:n_gam] == "_cell_angle_gamma":
            gamma = float(line.split()[1]) * deg2rad
            count_compl += 1
        if count_compl > 5.8:
            break

    v = sqrt(
        1
        - cos(alpha) ** 2
        - cos(beta) ** 2
        - cos(gamma) ** 2
        + 2 * cos(alpha) * cos(beta) * cos(gamma)
    )
    cell = np.array(
        [
            [a, 0, 0],
            [b * cos(gamma), b * sin(gamma), 0],
            [
                c * cos(beta),
                c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma),
                c * v / sin(gamma),
            ],
        ]
    )

    diag = np.diag(cell)
    nx, ny, nz = tuple(int(i) for i in np.ceil(cutoff / diag * 2.0))

    return nx, ny, nz


def create_input_file(template_path, output_path, **kwargs):
    # 템플릿 파일 읽기
    with open(template_path, "r") as file:
        template_content = file.read()

    # 템플릿 내용을 kwargs로 대체
    input_content = template_content.format(**kwargs)

    # 결과를 새로운 파일로 저장
    with open(output_path, "w") as file:
        file.write(input_content)

    print(f"Input file created: {output_path}")


from datetime import datetime
import random

def create_unique_directory(base_dir  ,pressure_point,temperature, adsorbate):
    # If the directory doesn't exist, create it directly
    # Get the current time with milliseconds
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Year-Month-Day Hour:Minute:Second:Milliseconds
    new_dir = f"{base_dir}_{adsorbate}_T{temperature}_P{pressure_point}_{current_time}_{random.randint(0, 1000000)}"

    # If the generated name already exists, append a counter
    counter = 1
    while os.path.exists(new_dir):
        new_dir = f"{base_dir}_{current_time}_{counter}"
        counter += 1

    os.makedirs(new_dir)
    return new_dir
def cropsim(targ_dir):
    dir_targ_nam = targ_dir

    basepath = os.getcwd()
    os.chdir(basepath)
    os.chdir(dir_targ_nam)
    os.chdir("Output/System_0")
    f_nam_list = os.listdir()
    # print(f_nam_list)

    prop_targ = "\tAverage loading absolute [mol/kg frame"
    n_prop_str = len(prop_targ)
    uptake_list = []
    for fn in f_nam_list:
        ff = open(fn)
        uptake_excess = -123.123
        # prop_targ = '\tAverage loading excess [mol/kg frame'
        ff_txt = ff.readlines()
        for ii in range(len(ff_txt)):
            targ_txt = "Finishing simulation"
            len_txt = len(targ_txt)
            if ff_txt[ii][:len_txt] == targ_txt:
                ff_txt_fin = ff_txt[ii:]
                break
        for txx in ff_txt_fin[::-1]:
            if txx[:n_prop_str] == prop_targ:
                txt_spl = txx.split()
                # print(txt_spl)
                # print(txt_spl[5])
                uptake_excess = float(txt_spl[5])
                uptake_list.append(uptake_excess)
    os.chdir(basepath)
    return uptake_list


def run_simulation_and_rename(sim_dir, raspa_dir):
    # 시뮬레이션 실행 코드
    simulate_exe = os.path.join(raspa_dir, "bin/simulate")
    os.chdir(sim_dir)
    os.system(f"{simulate_exe} simulation.input")
    base_dir = os.path.dirname(sim_dir)
    new_name = os.path.join(base_dir, "[complete]_" + os.path.basename(sim_dir))
    if os.path.exists(new_name):
        counter = 1
        unique_name = f"{new_name}_{counter}"
        while os.path.exists(unique_name):
            counter += 1
            unique_name = f"{new_name}_{counter}"
    else:
        unique_name = new_name
    os.rename(sim_dir, unique_name)
    print(f"Renamed directory: {sim_dir} -> {unique_name}")
    return new_name


# 비동기 함수로 정의
async def insert_GCMCResults_data(data):
    # GCMCResults 모델에 데이터 삽입
    new_result = GCMCResults(
        MoleculeName=data.MoleculeName,
        FrameworkName=data.FrameworkName,
        Uptake = data.Uptake,
        ExternalPressure=data.ExternalPressure,
        Temperature=data.Temperature,
        MoleculeDefinition=data.MoleculeDefinition,
        Forcefield=data.Forcefield,
        UnitCells=data.UnitCells,
        NumberOfCycles=data.NumberOfCycles,
        NumberOfInitializationCycles=data.NumberOfInitializationCycles,
        TranslationProbability=data.TranslationProbability,
        ReinsertionProbability=data.ReinsertionProbability,
        SwapProbability=data.SwapProbability,
        simulation_input_file=data.simulation_input_file,
        simulation_output_directory=data.simulation_output_directory,
        description=data.description,
        log_symbol=data.log_symbol,
        run_time = data.run_time
    )
    # 비동기적으로 저장
    await sync_to_async(new_result.save)()
    # new_result 객체 반환
    return new_result
# 비동기 함수로 정의

async def update_gcmc_results_with_isotherm(result_db_GCMCResults_record_list, result_db_IsothermFittingParameter_record):
    # 비동기 for문을 사용하여 각 GCMCResults 레코드 업데이트
    for gcmc_result in result_db_GCMCResults_record_list:
        # Isotherm_Fitting_Parameter 외래키에 result_db_IsothermFittingParameter_record 설정
        gcmc_result.Isotherm_Fitting_Parameter = result_db_IsothermFittingParameter_record
        
        # 비동기적으로 변경 사항 저장
        await sync_to_async(gcmc_result.save)()
async def insert_IsothermFittingParameter_data(data):
    # GCMCResults 모델에 데이터 삽입
    new_result = IsothermFittingParameter(
        MOF = data.MOF,
        MoleculeName = data.MoleculeName,
        Temperature = data.Temperature,
        Isotherm_model = data.Isotherm_model,
        K1 = data.K1,
        K2 = data.K2,
        # K3 = data.K3,
        # K4 = data.K4,
        error = data.error,
        optimizer = data.optimizer,
    )
    try:
        K3 = data.K3
        new_result["K3"] = K3 
    except:
        pass 
    try:
        K4 = data.K4
        new_result["K4"] = K4
    except:
        pass 

    # 비동기적으로 저장
    await sync_to_async(new_result.save)()
    # new_result 객체 반환
    return new_result
def gcmc_simulation(args,log_symbol):
    simulation_dir = args.simulation_dir
    pressure_point = args.pressure_point
    mof_cif_name = args.mof_cif_name
    cut_off = args.cut_off
    base_template_path = args.base_template_path
    force_field = args.force_field
    nunber_of_cycles = args.nunber_of_cycles
    nunber_of_initial_cycles = args.nunber_of_initial_cycles
    adsorbate = args.adsorbate
    temperature = args.temperature
    molecule_definition = args.molecule_definition
    translation_probability = args.translation_probability
    reinsertion_probability = args.reinsertion_probability
    swap_probability = args.swap_probability
    raspa_dir = args.raspa_dir
    description = args.description
    #########
    force_field_path = "%s/share/raspa/forcefield" % (raspa_dir)
    molecule_definition_path = "%s/share/raspa/molecules" % (raspa_dir)
    mof_path = "%s/share/raspa/structures/cif/" % (raspa_dir)
    ##
    mof_path = mof_path + mof_cif_name
    force_field_path = os.path.join(force_field_path, force_field)
    molecule_definition_path = os.path.join(
        molecule_definition_path, molecule_definition
    )
    #
    simulation_dir = args.simulation_dir
    mof_name = mof_cif_name.replace(".cif", "")
    input_dir = os.path.join(simulation_dir, f"inputs/{mof_name}")
    input_dir = create_unique_directory(input_dir,pressure_point,temperature, adsorbate)
    os.makedirs(input_dir, exist_ok=True)
    sim_input_file_path = os.path.join(input_dir, "simulation.input")
    ##
    ucell = cif2Ucell(mof_path, cut_off)
    ##
    params = {
        "MoleculeName": adsorbate,
        "FrameworkName": mof_name,
        "ExternalPressure": pressure_point,
        "Temperature": temperature,
        "MoleculeDefinition": molecule_definition,
        "Forcefield": force_field,
        "UnitCells": "{} {} {}".format(*ucell),
        "NumberOfCycles": nunber_of_cycles,
        "NumberOfInitializationCycles": nunber_of_initial_cycles,
        "TranslationProbability": translation_probability,
        "ReinsertionProbability": reinsertion_probability,
        "SwapProbability": swap_probability,
    }
    ## 실제 실행코드
    create_input_file(base_template_path, sim_input_file_path, **params)
    start_time = time.time()
    new_name = run_simulation_and_rename(input_dir, raspa_dir)
    run_time = time.time() - start_time
    print(new_name)
    uptake = cropsim(new_name)[0]

    ## 데이터베이스 저장코드
    data = params.copy() 
    data["simulation_input_file"] = os.path.join(new_name, "simulation.input")
    data["simulation_output_directory"] =  new_name
    data["Uptake"] = uptake 
    data["description"] = description
    data["log_symbol"] = log_symbol
    data["run_time"] = run_time
    result_db_record = asyncio.run(insert_GCMCResults_data(SimpleNamespace(**data)))
    return uptake,result_db_record


 #두 개의 초기 압력 포인트에 대해 시뮬레이션 실행
def simulate_initial_two_points(base_params,log_symbol):
    initial_pressure_points = ast.literal_eval(base_params["initial_pressure_points"])
    uptakes = { "pressure" : [] , "uptakes" : [] }
    result_db_record_list= []
    for initial_pressure in initial_pressure_points:
        params = {
            "simulation_dir": base_params["simulation_dir"],
            "pressure_point": initial_pressure,
            "mof_cif_name": base_params["mof_cif_name"],
            "cut_off": base_params["cut_off"],
            "base_template_path": base_params["base_template_path"],
            "force_field": base_params["force_field"],
            "nunber_of_cycles": base_params["nunber_of_cycles"],
            "nunber_of_initial_cycles": base_params["nunber_of_initial_cycles"],
            "adsorbate": base_params["adsorbate"],
            "temperature": base_params["temperature"],
            "molecule_definition": base_params["molecule_definition"],
            "translation_probability": base_params["translation_probability"],
            "reinsertion_probability": base_params["reinsertion_probability"],
            "swap_probability": base_params["swap_probability"],
            "raspa_dir": base_params["raspa_dir"],
            "description": base_params["description"],
        }
        # GCMC 시뮬레이션 실행
        params = SimpleNamespace(**params)
        uptake,result_db_record = gcmc_simulation(params,log_symbol)  # utils.gcmc_simulation 대신 직접 함수 호출
        uptakes["uptakes"].append( uptake )
        uptakes["pressure"].append(initial_pressure)
        result_db_record_list.append(result_db_record)
        # with open("a.txt", "w") as f:
        #     f.write(str(uptake))
    return uptakes,result_db_record_list