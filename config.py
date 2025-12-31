import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class DatasetConfig:
    """Configuration for each dataset"""
    name: str
    path: str
    label_column: str
    normal_label: str
    attack_types: Dict = field(default_factory=dict)
    train_test_split: bool = False
    
@dataclass
class ModelConfig:
    """Model hyperparameters"""
    n_estimators: int = 100
    max_depth: int = 10
    learning_rate: float = 0.1
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.1

class Config:
    """Main configuration class"""
    
    # Dataset paths with correct label column names
    DATASETS = {
        'nsl_kdd_train': DatasetConfig(
            name='NSL-KDD Train',
            path='KDDTrain.parquet',
            label_column='target',  # Changed from 'label' to 'target'
            normal_label='normal',
            attack_types={
                'DoS': ['neptune', 'smurf', 'back', 'teardrop', 'pod', 'land'],
                'Probe': ['satan', 'ipsweep', 'nmap', 'portsweep'],
                'R2L': ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy'],
                'U2R': ['buffer_overflow', 'loadmodule', 'rootkit', 'perl']
            }
        ),
        'nsl_kdd_test': DatasetConfig(
            name='NSL-KDD Test',
            path='KDDTest.parquet',
            label_column='target',  # Changed from 'label' to 'target'
            normal_label='normal',
            attack_types={
                'DoS': ['neptune', 'smurf', 'back', 'teardrop', 'pod', 'land', 'apache2', 'udpstorm', 'processtable', 'worm'],
                'Probe': ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint'],
                'R2L': ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy', 'named', 'snmpguess', 'xlock', 'xsnoop', 'sendmail', 'snmpgetattack'],
                'U2R': ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps', 'httptunnel']
            }
        ),
        'unsw_train': DatasetConfig(
            name='UNSW-NB15 Train',
            path='UNSW_NB15_training-set.parquet',
            label_column='attack_cat',  # Use attack_cat for UNSW
            normal_label='Normal',
            attack_types={
                'DoS': [],
                'Probe': ['Reconnaissance'],
                'R2L': ['Exploits', 'Generic', 'Shellcode', 'Worms'],
                'U2R': ['Backdoor', 'Analysis', 'Fuzzers']
            }
        ),
        'unsw_test': DatasetConfig(
            name='UNSW-NB15 Test',
            path='UNSW_NB15_testing-set.parquet',
            label_column='attack_cat',  # Use attack_cat for UNSW
            normal_label='Normal',
            attack_types={
                'DoS': [],
                'Probe': ['Reconnaissance'],
                'R2L': ['Exploits', 'Generic', 'Shellcode', 'Worms'],
                'U2R': ['Backdoor', 'Analysis', 'Fuzzers']
            }
        ),
        'cicids2017': DatasetConfig(
            name='CIC-IDS2017',
            path='cicids2017_cleaned.csv',
            label_column='Label',  # Capital L in CIC-IDS2017
            normal_label='BENIGN',
            attack_types={
                'DoS': ['DoS Hulk', 'DoS GoldenEye', 'DoS Slowloris', 'DoS Slowhttptest'],
                'DDoS': ['DDoS'],
                'Probe': ['PortScan'],
                'R2L': ['Bot', 'Infiltration', 'Web Attack � Brute Force', 'Web Attack � XSS', 'Web Attack � Sql Injection'],
                'U2R': ['FTP-Patator', 'SSH-Patator']
            },
            train_test_split=True
        )
    }
    
    # Unified labels for multi-dataset integration
    UNIFIED_LABELS = {
        0: 'Normal',
        1: 'DoS/DDoS',
        2: 'Probe/Scanning',
        3: 'Privilege Escalation',
        4: 'Malware/Exploit'
    }
    
    # Common feature names across datasets
    COMMON_FEATURES = [
        'duration', 'src_bytes', 'dst_bytes', 'protocol', 'service', 
        'flag', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'packet_length', 'packet_count', 'flow_duration', 'flow_bytes'
    ]
    
    # Paths
    RESULTS_DIR = 'results'
    MODELS_DIR = 'models'
    LOGS_DIR = 'logs'
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        for directory in [cls.RESULTS_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            os.makedirs(directory, exist_ok=True)