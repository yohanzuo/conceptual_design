'''
功能：计算旋翼诱导速度

by Yohan
'''

import numpy as np
import matplotlib
from matplotlib import pylab, mlab, pyplot
from numpy import *
from scipy import *
from scipy.integrate import dblquad, quad
from sympy import *
import math


'''
***********计算参量***********
syms r phi beta a0 a1 b1 beta_in dx dy phi_a AOA dT dL dM
r 桨叶径向位置；phi 方位角；beta=a0-a1*cos(phi)-b1*sin(phi) 挥舞角
beta_in 来流角；dx 基元阻力；dy 基元升力；phi_a 含操纵量的剖面安装角
AOA 剖面迎角；dT 基元拉力；dL 基元滚转力矩；dM 基元俯仰力矩
'''
r, phi, beta, a0, a1, b1, beta_in, dx, dy, phi_a, AOA, dT, dL, dM = 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0 ,0.0, 0.0, 0.0, 0.0,0.0, 0.0,   # symbols('r, phi,beta, a0, a1, b1, beta_in, dx,dy,phi_a, AOA, dT, dL, dM')
R = 5.25  # 桨叶半径
K = 3  # 桨叶片数
b7 = 0.35  # 0.7剖面弦长
D_phi = -7.3 * 180 / 3.1416  # 负扭转
phi_7 = 0  # 桨距
kappa = 0.95  # 叶端损失系数
I_ye = 38.3  # 桨叶挥舞惯性矩

omega = 40.44  # 旋翼转速
rho = 0.123  # 飞行状态下的空气密度
miu = 0.1422  # 前进比
lamda = -0.004  # 流入比
theta0 = 6.0435 * 180 / 3.1416  # 总距操纵配平数据
theta1 = 1.451 * 180 / 3.1416  # 横向周期变距操纵配平数据
theta2 = 1.4 * 180 / 3.1416  # 纵向周期变距操纵配平数据

Lock = 2.27  # 桨叶洛克数，Lock=5.731*b7*rho*R**4/(2*I_ye)
# 诱导速度计算
v0 = 3
vs = 3
vc = 3  # 初始化诱导速度
v0_pre = 0
vs_pre = 0
vc_pre = 0  # 初始化前一次迭代结果
error = 1  # 初始化最大迭代误差
n = 150  # 总迭代次数
accuracy = 0.001  # 精度
count = n  # 剩余迭代数

while (error > accuracy and count > 0):
    v1 = v0 + vs * r / R * sin(phi) + vc * r / R * cos(phi)  # 诱导速度，r与phi为变量
    a0 = kappa * Lock * (1 / 4 * (phi_7 + theta0) * (1 + miu ** 2) - 1 / 3 * (v0 / (omega * R) - lamda)
                         + 1 / 3 * miu * theta2 - 1 / 4 * miu * vs / (omega * R))
    a1 = (2 / 3 * (phi_7 + theta0) - 1 / 2 * (v0 / (omega * R) - lamda)) * 4 * miu / (1 - 0.5 * miu ** 2) \
         - 4 / 3 * vs / (omega * R) / (1 - 0.5 * miu ** 2) + theta2 * (1 + 2 * miu ** 2 / (1 - 0.5 * miu ** 2))
    b1 = (4 / 3 * (miu * a0 + vc / (omega * R)) / (1 + 0.5 * miu ** 2)) - theta1
    beta = a0 - a1 * cos(phi) - b1 * sin(phi)  # 挥舞角,phi为变量

    beta_in = ((v1 / (omega * R) - lamda) + r / R * (a1 * sin(phi) - b1 * cos(phi)) + miu * cos(phi) * beta) / (
        r / R + miu * sin(phi))  # 剖面来流角
    phi_a = phi_7 + theta0 + D_phi * (r / R - 0.7) + theta1 * cos(phi) + theta2 * sin(phi)  # 剖面安装角，r与phi为变量
    AOA = phi_a - beta_in  # 剖面迎角

    WW = (((v1 / (omega * R) - lamda) + r / R * (a1 * sin(phi) - b1 * cos(phi))
           + miu * cos(phi) * beta) ** 2 + (r / R + miu * sin(phi)) ** 2) * (omega * R) ** 2  # 周向与垂向来流合速度的平方，r与phi为变量

    alpha = AOA * 180 / 3.1416
    Cy = 0.12 * alpha + 0.088
    Cx = -2.5e-06 * alpha ** 4 + 3.6e-05 * alpha ** 3 - 3e-05 * alpha ** 2 - 0.0005 * alpha + 0.0062
    dy = 0.5 * rho * WW * b7 * Cy
    dx = 0.5 * rho * WW * b7 * Cx
    dT = (dy * cos(beta_in) - dx * sin(beta_in)) * cos(beta)



    dtt = lambda r, phi: dT
    dll = lambda r, phi: dT * sin(phi) * r
    dmm = lambda r, phi: dT * cos(phi) * r
    T = kappa * K / (2 * pi) * dblquad(dtt, 0.11, R, lambda phi: 0, lambda phi: 2 * pi)

    L = K / 2 * 10.92 * omega ** 2 * 0.11 * b1  # 滚转力矩 L=kappa*K/(2*pi)*dblquad(dll,0.11,R,0,2*pi)
    M = K / 2 * 10.92 * omega ** 2 * 0.11 * a1  # 俯仰力矩 M=kappa*K/(2*pi)*dblquad(dmm,0.11,R,0,2*pi)

    Ct = T / (0.5 * rho * omega ** 2 * pi * R ** 4)  # 拉力系数
    Cl = L / (0.5 * rho * omega ** 2 * pi * R ** 5)  # 滚转力矩系数
    Cm = M / (0.5 * rho * omega ** 2 * pi * R ** 5)  # 俯仰力矩系数

    V = ((lamda + v0 / (omega * R)) * (lamda + 2 * v0 / (omega * R)) + miu ** 2) / sqrt((lamda + v0 / (omega * R)) ** 2 + miu ** 2)
    a_L = arctan((lamda + v0 / (omega * R)) / miu)
    LL = 1 / V * [[0.5, 0, 15 * pi / 64 * sqrt(1 - sin(a_L) / (1 + sin(a_L)))], [0, -4 / (1 + sin(a_L)), 0],[15 * pi / 64 * sqrt((1 - sin(a_L)) / (1 + sin(a_L))), 0, -4 * sin(a_L) / (1 + sin(a_L))]]
    V_d = LL * np.array([Ct, Cl, Cm])  # 引入动态入流模型，使计算封闭
    v0 = V_d[0] * omega * R
    vs = V_d[1] * omega * R
    vc = V_d[2] * omega * R  # 更新诱导速度参量

    error = max(max(abs(v0 - v0_pre), abs(vs - vs_pre)), abs(vc - vc_pre))  # 迭代误差
    print('第', n - count + 1, '次迭代误差:', error)

    v0_pre = v0
    vs_pre = vs
    vc_pre = vc  # 更新前一次迭代结果
    count -= 1  # 剩余迭代数减1

if (error > accuracy):
    print('达到预定迭代次数，计算结束！！！')
else:
    print('满足精度要求，计算结束')
print('常量诱导速度系数v0:', v0 / (omega * R))
print('横向诱导速度系数vs:', vs / (omega * R))
print('纵向诱导速度系数vc:', vc / (omega * R))

# 参考文献：
#dynamic inflow 9th euro rotorcraft forum
#Flight dynamics simulation modeling for hingeless andbearingless rotor helicopters.
#电控旋翼直升机飞行动力学特性研究
#基于动态入流的无人直升机操纵特性研究
#直升机飞行动力学模型与飞行品质评估
