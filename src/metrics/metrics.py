import torch
import torch.nn.functional as F

##############################################
#   BASIC METRICS FOR REGRESSION (GLOBAL)
##############################################

def mae(y_pred, y_true):
    """
    Mean Absolute Error
    """
    return torch.mean(torch.abs(y_pred - y_true)).item()


def rmse(y_pred, y_true):
    """
    Root Mean Squared Error
    """
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()


def r2_score(y_pred, y_true):
    """
    Coefficient of Determination R²
    """
    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    return (1 - ss_res / ss_tot).item()


##############################################
#   METRICS PAR DIMENSION (PITCH, VELOCITY, DURATION)
##############################################

def mae_pitch(y_pred, y_true):
    return torch.mean(torch.abs(y_pred[:, 0] - y_true[:, 0])).item()


def mae_velocity(y_pred, y_true):
    return torch.mean(torch.abs(y_pred[:, 1] - y_true[:, 1])).item()


def mae_duration(y_pred, y_true):
    return torch.mean(torch.abs(y_pred[:, 2] - y_true[:, 2])).item()


def rmse_pitch(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred[:, 0] - y_true[:, 0]) ** 2)).item()


def rmse_velocity(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred[:, 1] - y_true[:, 1]) ** 2)).item()


def rmse_duration(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred[:, 2] - y_true[:, 2]) ** 2)).item()


##############################################
#   SIMPLE WRAPPER POUR L'AFFICHAGE GROUPÉ
##############################################

def metrics_dict():
    """
    Retourne un dict de métriques utilisables dans train(...)
    """
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2_score,
        'MAE_pitch': mae_pitch,
        'MAE_velocity': mae_velocity,
        'MAE_duration': mae_duration,
        'RMSE_pitch': rmse_pitch,
        'RMSE_velocity': rmse_velocity,
        'RMSE_duration': rmse_duration
    }
