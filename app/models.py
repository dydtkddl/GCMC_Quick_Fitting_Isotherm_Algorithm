from django.db import models

class IsothermFittingParameter(models.Model):
    MOF = models.CharField(max_length=100)  # MOF 프레임워크 이름
    MoleculeName = models.CharField(max_length=100)  # 흡착 분자 이름
    Temperature = models.FloatField()  # 온도 (예: 298.15 K)
    Isotherm_model = models.CharField(max_length=100)  # 사용된 이소텀 모델 이름
    K1 = models.FloatField(null=True, blank=True)  # K1 상수 값, None 허용
    K2 = models.FloatField(null=True, blank=True)  # K2 상수 값, None 허용
    K3 = models.FloatField(null=True, blank=True)  # K3 상수 값, None 허용
    K4 = models.FloatField(null=True, blank=True)  # K4 상수 값, None 허용
    error = models.FloatField(null=True, blank=True)  # K4 상수 값, None 허용
    optimizer = models.CharField(max_length=100,null=True, blank=True)  
    def __str__(self):
        return f"Isotherm Fitting: {self.MOF} with {self.MoleculeName} at {self.Temperature}K"

class GCMCResults(models.Model):
    MoleculeName = models.CharField(max_length=100)  # 흡착 분자의 이름
    Uptake = models.FloatField(default=123.123)  # 흡착량
    FrameworkName = models.CharField(max_length=100)  # MOF 프레임워크 이름
    ExternalPressure = models.FloatField()  # 외부 압력 (예: 0.5 bar)
    Temperature = models.FloatField()  # 온도 (예: 298.15 K)
    MoleculeDefinition = models.CharField(max_length=255)  # 분자 정의 파일 경로
    Forcefield = models.CharField(max_length=100)  # 포스필드 이름
    UnitCells = models.CharField(max_length=50)  # 단위 셀 정보 (예: "2 2 2" 형식)
    NumberOfCycles = models.IntegerField()  # 시뮬레이션 사이클 수
    NumberOfInitializationCycles = models.IntegerField()  # 초기화 사이클 수
    TranslationProbability = models.FloatField()  # 이동 확률
    ReinsertionProbability = models.FloatField()  # 재삽입 확률
    SwapProbability = models.FloatField()  # 교환 확률
    simulation_input_file = models.TextField()  # 시뮬레이션 입력 파일 경로
    simulation_output_directory = (
        models.TextField()
    )  # 시뮬레이션 출력 파일 내용 또는 경로
    created_at = models.DateTimeField(auto_now_add=True)  # 레코드가 생성된 시간
    updated_at = models.DateTimeField(auto_now=True)  # 레코드가 수정된 시간
    description = models.CharField(max_length=200, default = "no description")  # 단위 셀 정보 (예: "2 2 2" 형식)
    log_symbol = models.CharField(max_length=200, default = "log_symbol")  # 단위 셀 정보 (예: "2 2 2" 형식)
    run_time = models.CharField(max_length=200, default = "run_time")  # 단위 셀 정보 (예: "2 2 2" 형식)
    Isotherm_Fitting_Parameter = models.ForeignKey(IsothermFittingParameter, on_delete=models.CASCADE, null=True, blank=True)  # 외래키 설정, None 허용
    def __str__(self):
        return f"Simulation Result: {self.FrameworkName} with {self.MoleculeName}"
