# -*- coding=utf-8 -*-
"""
功能：根据直升机的参数计算直升机的飞行性能
输入：直升机重量
      旋翼和尾桨尺寸
      发动机参数
      大气参数
      *作为输入的参数已在程序开头列出，参数值为样例直升机（UH-60A）的参数。常规大气环境不需要输入环境参数，特殊飞行环境
      可以输入环境参数进行计算。

输出：有地效悬停升限
      无地效悬停升限
      最大垂直爬升率
      最大速度
      巡航速度
      经济速度
      航程
      航时

两个重要的图：1）功率——前飞速度图（延伸：高度——最大速度图）函数forwardflight（ line25: plt.show）
              2）高度——垂直爬升率图
              *详细意义见参考文献[2]p260-261

说明：
1）程序主要的计算的过程和逻辑参见参考文献:[1]p77, [3]p118, [4]p207 ;
2) 过程中遇到的基础知识查阅参考文献[2],[3],[6]
3）飞行环境参见文献[6]
4) 报告的书写以及后续界面的开发，参见文献[5],但是慎用文献[5]中的参数,很多是英制单位的，外文文献单位一定要注意；
   界面的交互逻辑可以参考文献[5]和Advanced Aircraft Analysis软件

运行要求：python3.x 包含库：numpy，matplotlib

#参考文献：
#[1] 直升机总体多学科优化设计，彭明华
#[2] Rotorcraft Aeromechanics，Wayne Johnson
#[3] 直升机空气动力学，王适存
#[4] 直升机性能及稳定性和操纵性
#[5]DESIGN METHODOLOGY FOR DEVELOPING CONCEPT INDEPENDENT ROTORCRAFT ANAYSIS AND DESIGN SOFTWARE
#[6]飞机设计手册，第1册，第19册

by Yohan
"""


import scipy as sp
import numpy as np
from scipy import *
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

font1 = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    #定义字体
font2 = {'family':'SimHei','size':8}
pi = np.pi

############# 输入：直升机参数(默认值UH-60参数)

### 发动机参数（ISA）
N_eng=2        #发动机个数
p_eng=1209    #发动机额定功率,kw
zeta_eng=0.9   #发动机安装功率损失
sfc=0.29

### 几何参数
# 主旋翼参数
r_mr = 8.1778  # 主旋翼半径，m
r_root = 0.09 * r_mr  # 桨叶根切
chord_mr = 0.527  # 主旋翼弦长
nb_mr = 4  # 主旋翼桨叶片数
v_tip_mr = 221  # 主旋翼桨尖速度,m/s
omega_mr = v_tip_mr / r_mr  # 主旋翼转速，rad/s
sigma_mr = nb_mr * r_mr * chord_mr / (pi * (r_mr ** 2 - r_root ** 2))  # 主旋翼实度

# 尾桨参数
r_tr = 1.6764
r_root_tr = 0.04 * r_tr
chord_tr = 0.247
nb_tr = 4
v_tip_tr = 208
omega = v_tip_tr / r_tr
sigma_tr = nb_tr * r_tr *chord_tr/ (pi * (r_tr ** 2 - r_root_tr ** 2))

# 其他参数
l_mtot = 0.0721  # 旋翼桨尖到尾桨桨尖间距
l_arm = r_mr + r_tr + l_mtot  # 旋翼轴到尾桨轴间距（直接输入或计算）
DL = 45.4  # 桨盘载荷,kg/m^2
f = 3  # 全机废阻系数(当量阻力平板面积,m^2）
zeta_acc = 0.05  # 附件功率比例
zeta_vt = 1.15  # 垂尾干扰修正系数
epsilon = 0.9  # 直升机功率利用系数

# 重量参数
W_DG = 9200  # 设计总重,kg
W_MTO = 10660  # 最大起飞重量,kg
W_empty = 4950 # 空机重量（总重的比例）,kg
W_ful = 1060.5  # 燃油重量,kg
W_pay = 1246  # 有效载荷,kg
W_ful_unav = 100 # 滑油和不可用燃油重量，kg
W_ful_av=W_ful-W_ful_unav  #可用燃油

###### 大气环境参数
def ambient_conditions(h):
    H = 6356766 * h / (6356766 + h)  # 位势高度
    if 0<=H <= 11000:
        temperature = 15 - 0.0065 * H   #温度，摄氏度
        pressure = 1.01325 * 10 ** 5 * (1 - 0.225577 * 10 ** (-4) * H) ** 5.25588   # 压强，Pa
        density = 1.225 * (1 - 0.225577 * 10 ** (-4) * H) ** 4.25588   # 密度，kg/m^3
    elif 11000<H<20000:
        temperature = -56.5
        pressure = 2.263204 * 10 ** 4 * np.exp(- 0.225577 * 10 ** (-4) * (H - 11000))
        density = 0.3639176 * np.exp(- 0.225577 * 10 ** (-4) * (H - 11000))
    else:
        print('高度范围0-20000，请重新输入')
    v_a = 20.0468 * sqrt(temperature + 273)  #声速，m/s
    g = 9.80665 * (6356766 / (6356766 + h)) ** 2  # 重力加速度，kg*m/s^2
    amb=(temperature, pressure, density, g, v_a)
    return amb

# 翼型参数
a1 = np.array([0.1, 0, 0.01, 0, 0, 0, 0])
a2 = np.array([0.1, 0, 0.01, 0, 0, 0, 0])
cl = lambda alpha: a1[0] * alpha + a1[1]    # 升力系数
cd = lambda alpha: a1[2] + a1[3] * alpha + a1[4] * alpha ** 2 + a1[5] * alpha ** 3 + a1[6] * alpha ** 4  # 阻力系数
cl_tr = lambda alpha_t: a2[0] * alpha_t + a2[1]
cd_tr = lambda alpha_t: a2[2] + a2[3] * alpha_t + a2[4] * alpha_t ** 2 + a2[5] * alpha_t ** 3 + a2[6] * alpha_t ** 4
cl_max=1.2  #0.7R处最大升力系数


# 计算需用功率
def power_rq(h, v_1):    # h为海拔高度，v_1为前飞速度
    amb=ambient_conditions(h)  # 大气参数
    rho = amb[2]
    g = amb[3]
    D = 0.5 * rho * v_1 ** 2 * f  # 全机阻力
    k_v=1+DL/1000   #垂直增重系数
    T = np.sqrt(D * D + (W_DG * g * k_v)**2)  # 前飞需用拉力
    C_t = 2 * T / (rho * pi * r_mr * r_mr * v_tip_mr ** 2)   # 拉力系数
    mu = v_1 / v_tip_mr  # 前进比
    k_t=0.98*(1-mu**2)  #拉力修正系数
    alpha = 3 * C_t* k_t / (sigma_mr * a1[0])   # 桨叶平均迎角
    alpha_s = -(60 * f * mu * mu / (pi * r_mr * r_mr * C_t) + 105 * C_t * mu / sigma_mr) / 57.3   # 桨盘迎角

    # 计算诱导速度
    B = 1 - np.sqrt(2 * C_t) / nb_mr   # 桨尖损失修正系数（一般可取固定值0.97）
    v_i0b = 0.5 * np.sqrt(C_t/ B)   # 特性诱导速度
    lambda_i = 1   # 诱导速度比值
    lambda_mu = mu / v_i0b
    for i in range(1, 200):
        lambda_i -= ((lambda_i ** 4 - 2 * lambda_i ** 3 * lambda_mu * sin(alpha_s) + (lambda_i * lambda_mu) ** 2 - 1) /
                     (4 * lambda_i ** 3 - 6 * lambda_i ** 2 * lambda_mu * sin(alpha_s) + 2 * lambda_i * lambda_mu ** 2))
    v_ib = lambda_i * v_i0b  # 前飞诱导速度

    # 计算功率
    J_0 = 1.07  # 悬停诱导功率修正系数，1.05-1.1
    J = J_0 * (1 + 3 * mu ** 2)  # 前飞诱导功率修正系数
    K_p0 = 1.05  # 悬停型阻功率修正系数，1.05-1.1
    P_induced = 0.5 * rho * pi * r_mr * r_mr * v_tip_mr ** 3 * J * C_t * v_ib / 1000  # 诱导功率,kw
    P_profile = 1 / 8 * rho * pi * r_mr * r_mr * v_tip_mr ** 3 * K_p0 * (1 + 4.65 * mu * mu) * sigma_mr * cd(alpha)/1000  # 型阻功率,kw
    P_parasite = 0.5 * rho * f * v_1 ** 3/1000  # 废阻功率,kw
    P_mr = P_induced + P_profile + P_parasite  # 主旋翼功率,kw

    Q_mr = P_mr*1000 / omega_mr  # 反扭矩
    T_tr = zeta_vt * Q_mr / l_arm  # 尾桨拉力
    mu_tr = v_1/v_tip_tr
    J_tr = 1.1* (1 + 3 * mu_tr ** 2)
    C_t_tr = 2 * T_tr / (rho * pi * r_tr * r_tr * v_tip_tr ** 2)
    K_p0_tr = 1.1
    alpha_t = -3 * C_t_tr / (sigma_tr * a2[0])
    v_i0_tr=0.5*np.sqrt(C_t_tr/0.92)
    lambda_i_tr = 1
    lambda_mu_tr = mu_tr / v_i0_tr
    for i in range(1, 200):
        lambda_i_tr -= ((lambda_i_tr ** 4 +(lambda_i_tr * lambda_mu_tr) ** 2 - 1) / (4 * lambda_i_tr ** 3 + 2 * lambda_i_tr * lambda_mu_tr ** 2))
    v_i_tr=lambda_i_tr*v_i0_tr
    P_tr_induced = 0.5 * rho * pi * (r_tr ** 2) * (v_tip_tr ** 3) * J_tr * C_t_tr * v_i_tr/1000
    P_tr_profile = 1 / 8 * rho * pi * (r_tr ** 2) * (v_tip_tr ** 3) * K_p0_tr * (1+4.6 * mu_tr * mu_tr) * sigma_tr * cd(alpha_t)/1000
    P_tr = P_tr_induced+P_tr_profile  # 尾桨需用功率

    P_acc = zeta_acc * (P_mr + P_tr)   #附件功率和传动损失
    P_total = P_mr + P_tr + P_acc  # 需用功率=旋翼功率+尾桨功率+附件功率和传动损失
    return P_total, P_mr, P_tr, P_induced, P_profile, P_parasite


# 计算发动机可用功率
def power_av(h):
    p_av_ISA = N_eng*p_eng
    amb = ambient_conditions(h)
    t_air = amb[0]
    P_av=zeta_eng*p_av_ISA*(1-0.195*h*3.28/10000)*(1-0.009*(t_air-15))
    return P_av

# 计算垂直飞行性能
def verticalflight():
    # 悬停升限（有地效/无地效）
    h_v=np.arange(0,10000,5)  #初值，不需要输入
    mu = 0
    h_IGE=1.4  #地效高度z/R
    f_IGE =(0.146 + 2.090*h_IGE - 2.068*h_IGE**2+ 0.932*h_IGE**3 - 0.157*h_IGE**4)**(-2/3)
    #f_IGE =(0.9926+0.3794/(2*h_IGE)**2)**(2/3)   # f_IGE多种算法见参考文献[2],p120
    vy=np.zeros(np.alen(h_v))  #创建数组存放v_y(注意长度，为了作图）
    for h in range(0,10000,5):
        amb2 = ambient_conditions(h)
        rho=amb2[2]
        g=amb2[3]
        tem=power_rq(h,mu)
        delta_P = power_av(h) - tem[0]   # 剩余功率
        delta_P1=power_av(h)-(tem[2]+tem[3]*f_IGE+tem[4])
        v_i0 = 0.95*np.sqrt(W_DG*g/(2*rho*pi*r_mr**2))
        v_y1 = 1000 * epsilon * delta_P / (W_DG*g)
        k_vy = 1 + 1 / (1 + v_y1 / v_i0)   # 垂直飞行时诱导速度修正系数
        v_y = v_y1 * k_vy
        if 0<v_y<0.5:
            h1=h
        i=h//5
        vy[i] = v_y
        v_y1_IGE = 1000 * epsilon * delta_P1 / (W_DG*g)
        k_vy_IGE = 1 + 1 / (1 + v_y1_IGE / v_i0)
        v_y_IGE = v_y1_IGE * k_vy_IGE
        if 0<v_y_IGE<0.5:
            h2=h
            print('无地效悬停升限=', h1)
            print('有地效悬停升限=', h2)
            break

    # 垂直爬升率
    v_y_max=np.ma.max(vy)
    print('最大垂直爬升率=',v_y_max)
    plt.plot(vy[:h1//5+1],h_v[:h1//5+1])
    plt.xlabel('最大垂直爬升率Vy(m/s)', fontproperties=font1)
    plt.ylabel('高度H(m)', fontproperties=font1)
    plt.title('高度—垂直爬升率 曲线', fontproperties=font1)
    plt.show()

verticalflight()


#计算前飞性能
def forwardflight():
    v = np.arange(0, 110, 1)   #用数组代替循环
    for h in range(0, 10000, 500):   #对高度进行迭代，计算不同高度飞行性能，其中巡航高度通常设置为1000m
        print(h, '米高度下的飞行性能：')
        ### 最大前飞速度
        ## 功率限制最大速度v_max1
        # 不同高度下的功率-速度曲线
        P_total, P_mr, P_tr, P_induced, P_profile, P_parasite = power_rq(h, v)
        P_av = power_av(h)*np.ones(np.alen(v))
        P_rq=np.array(P_total)
        cof1=np.polyfit(v,P_rq,6)  # 多项式拟合得到功率曲线
        p1=np.poly1d(cof1)
        # plt.plot(v,p1(v))
        plt.figure(1)    # 创建绘图
        line1 = plt.plot(v * 3.6, P_rq)
        line2 = plt.plot(v * 3.6, P_mr)
        line3 = plt.plot(v * 3.6, P_tr)
        line4 = plt.plot(v * 3.6, P_induced)
        line5 = plt.plot(v * 3.6, P_profile)
        line6 = plt.plot(v * 3.6, P_parasite)
        line7 = plt.plot(v * 3.6, P_av)
        plt.xlabel('前飞速度V(m/s)', fontproperties=font1)
        plt.ylabel('功率P(kw)', fontproperties=font1)
        plt.title('前飞需用功率—速度 曲线', fontproperties=font1)
        plt.legend(['总需用功率', '旋翼功率','尾桨功率','诱导功率','型阻功率',"废阻功率",'可用功率'], loc=9, prop=font2)
        plt.show()  # 显示功率——前飞速度图
        delta_P3 = P_av - P_rq
        cof=np.polyfit(v,delta_P3,7)
        p=np.poly1d(cof)
        # plt.plot(v, delta_P3, 'o', v, p(v), lw=2)
        v_roots=np.roots(p)
        v_roots_real=isreal(v_roots)*v_roots
        for i in range(0,np.alen(v_roots),1):
            if 0<v_roots_real[i]<=110:
                v_max_1 = 3.6*(v_roots_real[i]).real
                break

        # 后行桨叶气流分离限制最大速度v_max2
        amb2 = ambient_conditions(h)
        rho = amb2[2]
        v_sonic = amb2[4]
        g=amb2[3]
        k_t = 0.98
        v_max_2=3.6*rho*(v_tip_mr**3)*0.92*sigma_mr*k_t*cl_max/(24*DL*9.8)-v_tip_mr/4

        # 前行桨叶激波限制最大速度v_max3
        v_max_3=3.6*(0.9*v_sonic-v_tip_mr)
        v_max=min(v_max_1,v_max_2,v_max_3)
        print('最大前飞速度', 'v_max=', v_max)

        # 巡航速度和最大航程
        p2 = np.polyder(p1)
        for x0 in range(1, 1100, 1):
            x=x0/10
            if abs(p2(x)-p1(x)/x)<0.1:
                v_cruise=x/0.9
                print('巡航速度：',3.6*v_cruise)
                L=v_cruise*3.6*W_ful_av/(sfc*p1(v_cruise))+12  # 航程，单位km
                print('航程：', L, '千米')
                break
        # 最大续航时间和经济速度
        delta_t1=6/60  #起飞时间，6-8分钟
        delta_t2=6/60  #着陆时间，6-8分钟
        for x1 in range(0,1100,1):
            x2=x1/10
            if -5<p1(x2)-min(P_rq)<0:
                v_jj=x2  #经济速度，m/s
                print('经济速度：',3.6*v_jj,'km/h')
                t=W_ful_av/(sfc*p1(v_jj))+delta_t1+delta_t2   # 最大续航时间,单位：小时
                print('航时：',t,'h')
                # 最大前飞爬升率
                # 经济速度时剩余功率最多，用于爬升的功率最大
                for k in range(0,1000,1):
                    zeta_profile=1.1  # 型阻功率修正
                    delta_P4=zeta_profile*p(v_jj)
                    v_f1=1000*0.9*zeta_eng*delta_P4/(W_DG*g)
                    v1=np.sqrt(v_f1*v_f1+v_jj*v_jj)
                    delta_P5 = zeta_profile*p(v1)
                    v_f2=1000*0.9*zeta_eng*delta_P5/(W_DG*g)
                    if v_f1>0 and v_f2>0 and abs(v_f1-v_f2)<0.1:
                        v_f=v_f1
                        print('最大斜爬升率：',v_f)
                        break
                    v_jj=v1
                break

forwardflight()