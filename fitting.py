import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error
def langmuir(K1, K2, P):
    """Langmuir isotherm model"""
    q = (K1 * K2 * P) / (1 + K2 * P)
    return q
def freundlich(K1, K2, P):
    """Freundlich isotherm model"""
    q = K1 * P**K2
    return q
def sips(K1, K2, K3, P):
    """Sips isotherm model"""
    q = (K1 * K2 * P**K3) / (1 + K2 * P**K3)
    return q
def quadratic(K1, K2, K3, P):
    """Quadratic isotherm model"""
    q = (K1 * (K2 * P + K3 * P**2)) / (1 + K2 * P + 2 * K3 * P**2)
    return q
def peleg(K1,K2, K3, K4, P):
    """Peleg isotherm model"""
    q = K1 * P**K2 + K3 * P**K4
    return q


# 여러 최적화 기법 리스트
OPTIMIZERS = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg']
# 초기 추정 파라미터 (Langmuir용: [K1, K2], Freundlich용: [K1, K3])
def estimate_langmuir_parameters(P, q):
    q1 ,q2 = q 
    P1, P2 = P 
    K2 = (q1 / q2 * P2 - P1) / (P1 * P2 * (1 - q1 / q2))
    K1 = (q1 * (1 + K2 * P1)) / (K2 * P1)
    return K1, K2

def objective_function(params, model_func, P, q_exp, error_metric="MSE"):
    """
    Objective function for optimization.
    
    error_metric: "MSE", "MAE", or "NRMSE"
    params: list of parameters to optimize [K1, K2, ...]
    model_func: the isotherm model function (e.g., langmuir, freundlich)
    P: pressure data (array)
    q_exp: experimental adsorption data (array)
    """
    # 모델로 예측한 흡착량 계산
    q_pred = model_func(*params, P)
    # 선택된 에러 메트릭에 따라 계산
    if error_metric == "MSE":
        error = mean_squared_error(q_exp, q_pred)
    elif error_metric == "MAE":
        error = mean_absolute_error(q_exp, q_pred)
    elif error_metric == "NRMSE":
        mse = mean_squared_error(q_exp, q_pred)
        rmse = np.sqrt(mse)
        error = rmse / (np.max(q_exp) - np.min(q_exp))  # NRMSE 계산
    else:
        raise ValueError("Unsupported error metric. Choose from 'MSE', 'MAE', or 'NRMSE'.")
    
    return error
def fit_model_with_optimizers(P, q_exp, model_name, error_metric="MSE"):
    if not isinstance(P, np.ndarray):
        P = np.array(P)
    if not isinstance(q_exp, np.ndarray):
        q_exp = np.array(q_exp)
    best_result = None
    best_optimizer = None
    best_error = float('inf')  # 높은 값으로 시작
    model_func = ""
    # 모델 선택 및 초기 추정값 설정
    if model_name == "langmuir":
        initial_guess = np.array([1.0, 1.0])
        model_func = langmuir
    elif model_name == "freundlich":
        initial_guess = np.array([1.0, 1.0])
        model_func = freundlich
    elif model_name == "sips":
        initial_guess = np.array([1.0, 1.0, 1.0])
        model_func = sips
    elif model_name == "quadratic":
        initial_guess = np.array([1.0, 1.0, 1.0])
        model_func = quadratic
    elif model_name == "peleg":
        initial_guess = np.array([1.0, 1.0, 1.0, 1.0])
        model_func = peleg
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # 여러 최적화 기법을 시도
    for optimizer in OPTIMIZERS:
        try:
            # 최적화 실행
            result = minimize(objective_function, initial_guess, args=(model_func, P, q_exp, error_metric), method=optimizer)
            error = result.fun  # 계산된 에러 값 (MSE, MAE, NRMSE 중 선택)

            # 최적의 에러 메트릭 값 업데이트
            if error < best_error:
                best_error = error
                best_result = result
                best_optimizer = optimizer
        except Exception as e:
            print(f"Optimizer {optimizer} failed: {e}")
    
    return best_result, best_optimizer
