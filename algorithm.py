import pandas as pd
from types import SimpleNamespace
import os
import argparse
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import utils
import fitting
import determine_next_step
import numpy as np
#
import logging
import time 
import asyncio

from datetime import datetime 

def setup_logging(log_dir, idx, start_time):
    """로그 설정 함수 (병렬 처리에서 각 프로세스마다 로그 설정)"""
    # 현재 시간 및 작업 번호 기반으로 로그 파일 이름 생성
    
    log_file_name = f"log_{start_time}_task_{idx}.log"
    
    # 로그 파일 전체 경로
    log_file_path = os.path.join(log_dir, log_file_name)

    # 디렉토리가 존재하지 않으면 생성
    os.makedirs(log_dir, exist_ok=True)

    # 로그 설정
    logging.basicConfig(
        filename=log_file_path, 
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print(f"Logging to {log_file_path}")

def log_and_store(log_messages, message, level="info"):
    """로그 메시지를 저장하고 출력하는 함수"""
    log_messages.append(message)

    if level == "info":
        logging.info(message)
    elif level == "error":
        logging.error(message)

def main():
    # ArgumentParser 설정
    parser = argparse.ArgumentParser()
    log_symbol = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument("-csv", "--input_csv", type=str, help="Path to input CSV file", required=True)
    parser.add_argument("-num_cpus", "--num_cpus", type=int, help="Number of CPUs to use", default=cpu_count())
    parser.add_argument("-log", "--log_file_path", type=str, help="log_file_path")
    args = parser.parse_args()
    
    # CSV 파일 읽기
    df = pd.read_csv(args.input_csv)

    # 각 행을 딕셔너리로 변환하여 병렬 처리에 사용할 리스트 생성
    sim_inputs = [row.to_dict() for _, row in df.iterrows()]
    
    # 로그 기록: 작업 시작
    log_and_store([], f"Starting parallel simulation with {len(sim_inputs)} inputs.")

    # 병렬 처리로 algorithm 함수 실행, log_file_path 전달
    Parallel(n_jobs=args.num_cpus)(
        delayed(algorithm)(SimpleNamespace(**args_dict), idx, len(sim_inputs), args.log_file_path,log_symbol) 
        for idx, args_dict in enumerate(sim_inputs)
    )
def fit_func(model_name):
    if model_name == "langmuir":
        return fitting.langmuir
    if model_name == "freundlich":
        return fitting.freundlich
    if model_name == "sips":
        return fitting.sips
    if model_name == "quadratic":
        return fitting.quadratic
    if model_name == "peleg":
        return fitting.peleg

def algorithm(args, idx, total_tasks, log_file_path, log_symbol):
    # 각 작업에 대한 로그 메시지 저장용 리스트
    log_messages = []
    # 로그 설정
    setup_logging(log_file_path, idx,log_symbol)
    # 작업 시작 시 로그 기록
    log_and_store(log_messages, f"Task {idx + 1}/{total_tasks} started: {args.adsorbate}, {args.mof_cif_name}, {args.temperature}K")
    task_start_time = time.time()
    # 기본 시뮬레이션 파라미터 설정
    base_params = {
        "simulation_dir": args.simulation_dir,
        "initial_pressure_points": args.initial_pressure_points,
        "mof_cif_name": args.mof_cif_name,
        "cut_off": args.cut_off,
        "base_template_path": args.base_template_path,
        "force_field": args.force_field,
        "nunber_of_cycles": args.nunber_of_cycles,
        "nunber_of_initial_cycles": args.nunber_of_initial_cycles,
        "adsorbate": args.adsorbate,
        "temperature": args.temperature,
        "molecule_definition": args.molecule_definition,
        "translation_probability": args.translation_probability,
        "reinsertion_probability": args.reinsertion_probability,
        "swap_probability": args.swap_probability,
        "raspa_dir": args.raspa_dir,
        "description": args.description
    }
    try_count = 0
    # try:
    # 두 개의 초기 압력 포인트로 시뮬레이션 수행
    # 두 개의 초기 압력 포인트로 시뮬레이션 시작 전 시간 기록
    initial_sim_start_time = time.time()
    log_and_store(log_messages,f"Task {idx + 1} - Initial two points simulation started: Adsorbate = {args.adsorbate}, MOF = {args.mof_cif_name}, Temperature = {args.temperature}K, Initial Pressures = {args.initial_pressure_points}")
    uptakes,result_db_GCMCResults_record_list = utils.simulate_initial_two_points(base_params,log_symbol)
    # 시뮬레이션 완료 후 시간 기록 및 경과 시간 계산
    initial_sim_end_time = time.time()
    initial_duration = initial_sim_end_time - initial_sim_start_time
    log_and_store(log_messages,f"Task {idx + 1} - Initial two points simulation completed in {initial_duration:.2f} seconds.")
    new_P = np.array(uptakes["pressure"]) / 1e5
    new_uptakes = np.array(uptakes["uptakes"])

    try_count = 3
    
    upper_count = 2
    pressure_upper_list = [0.5, 1.5, 7, 12, 15, 18]
    
    while upper_count < 6 and try_count < 13:
        loop_start_time = time.time()
        ########## uptakes로 Fitting해보기 2개로만
        MODEL_NAMES = ["langmuir", "freundlich", "sips", "quadratic", "peleg"]
        
        ## 다음 P 결정
        flag_list = []
        next_p = -1
        
        for i in range(len(new_uptakes) - 1):
            qs, ps, sats, qm = determine_next_step.calculate_saturation_factor(new_uptakes[i:i+2], new_P[i:i+2])
            flag = sum(sats >= 0.9)
            flag_list.append(flag)
            # log_and_store(log_messages,f"{str(sats)}")
        if sum(flag_list) == 0:
            next_p = pressure_upper_list[upper_count]  # 더 전진
            upper_count += 1
        else:
            if 1 in flag_list:
                next_p = np.mean(new_P[flag_list.index(1):flag_list.index(1) + 2])
            else:
                if (len(flag_list)) >=2:
                    next_p = np.mean(new_P[flag_list.index(2) - 1:flag_list.index(2) + 1])
        next_p *= 1e5
        
        log_and_store(log_messages,f"Task {idx + 1} - Next pressure point: {next_p:.2f} Pa")

        ## 결정한 다음 P로 시뮬레이션 진행
        params = base_params.copy()
        params["pressure_point"] = next_p
        params = SimpleNamespace(**params)
        uptake,result_db_GCMCResults_record = utils.gcmc_simulation(params,log_symbol)
        result_db_GCMCResults_record_list.append(result_db_GCMCResults_record)
        if uptake == 0 : 
            break
        # 시뮬레이션 완료 시간 기록
        loop_end_time = time.time()
        loop_duration = loop_end_time - loop_start_time
        log_and_store(log_messages,f"Task {idx + 1} - Iteration completed in {loop_duration:.2f} seconds. Next pressure = {next_p:.2f} Pa")


        old_P = new_P.copy()
        old_uptakes = new_uptakes.copy()
        new_P = np.append(new_P, next_p/1e5)
        new_uptakes = np.append(new_uptakes, uptake)

        Model_list = []
        NRMSE_VALUE = {}

        if len(new_P) == 2:
            Model_list = MODEL_NAMES[:2]
        elif len(new_P) == 3:
            Model_list = MODEL_NAMES[:4]
        else:
            Model_list = MODEL_NAMES
        old_fitting_list = []
        min_error = float('inf')
        best_model = None
        best_params = None
        best_optimizer = None
        for model in Model_list:
            old_fitting = fitting.fit_model_with_optimizers(old_P, old_uptakes, model, error_metric="NRMSE")
            if old_fitting[0] is not None:
                error = fitting.objective_function(old_fitting[0].x, fit_func(model), new_P, new_uptakes, error_metric="NRMSE")
                NRMSE_VALUE[model] = error
                if error < min_error:
                    min_error = error
                    best_model = model
                    best_params = old_fitting[0].x
                    best_optimizer = old_fitting[1]
        log_and_store(log_messages, f"#### {try_count}th Try \tmin_error : {min_error} \ttest_model : {best_model} \tbest_param : {str(best_params)}")
        log_and_store([], f"Fitting With {str(old_P)} / {str(old_uptakes)} && Evaluate With {str(new_P)} / {str(new_uptakes)}")
        if np.sum(np.array(list(NRMSE_VALUE.values())) <= 0.03) >= 1:
            log_and_store(log_messages,f"Task {idx + 1} - Fitting complete for {args.adsorbate}, {args.mof_cif_name}. Ending process.")
            break
        try_count += 1
    log_and_store(log_messages,f"Task {idx + 1} - Process ended: upper_count = {upper_count}, try_count = {try_count}")
    log_and_store([], f"Pressure Points : {str(new_P)} / Uptake Points : {str(new_uptakes)}")
    ## 
    # , best_params,,,,, 를 활용
    FittingData = {
        "MOF" :   args.mof_cif_name   ,
        "MoleculeName" :    args.adsorbate  ,
        "Temperature" : args.temperature     ,
        "Isotherm_model" :   best_model   ,
                "error" :  min_error    ,
        "optimizer" :  best_optimizer    ,
    }
    for ii , pa in enumerate(best_params):
        FittingData["K%s"%(ii+1)] = pa 
    result_db_IsothermFittingParameter_record = asyncio.run(utils.insert_IsothermFittingParameter_data(SimpleNamespace(**FittingData)))

    # 예시로 asyncio.run()을 통해 비동기 함수 호출
    asyncio.run(utils.update_gcmc_results_with_isotherm(result_db_GCMCResults_record_list, result_db_IsothermFittingParameter_record))
    # 총 작업 시간 기록
    task_end_time = time.time()
    total_task_duration = task_end_time - task_start_time
    log_and_store(log_messages,f"Task {idx + 1} completed in {total_task_duration:.2f} seconds.")
    # 완료 후 전체 로그를 순차적으로 다시 출력
    log_and_store(log_messages,f"\n===== Full log for Task {idx + 1} =====\n" + "\n".join(log_messages))
    # except Exception as e:
    #     # 작업 실패 시 로그 기록
    #     logging.error(f"Task {idx + 1}/{total_tasks} failed: {str(e)}\ttry_count = {try_count}")

        
if __name__ == "__main__":
    main()
