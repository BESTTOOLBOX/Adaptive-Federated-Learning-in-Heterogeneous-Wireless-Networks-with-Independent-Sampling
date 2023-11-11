from math import log, exp
import copy
import csv
import matlab.engine

class optimize():
	def __init__(self,myid,num,total_B,delta_H,H0,S,N0,m): #Eres是当前轮次各用户剩余能量, list
		self.id=myid #程序执行id, string
		self.num=num #总用户数, int
		self.total_B=total_B #总带宽, float
		self.delta_H=delta_H #delta_H, float
		self.H0=H0	#H0, int
		self.S=S #模型大小, float
		self.alpha=[[] for r in range(0,num)] #alpha记录用于求和, list套list
		self.N0=N0 #噪声, float
		self.m=m #每轮次选取的设备数量
		self.solved_b=[]
		self.solved_e=[0 for i in range(0,self.num)]
		self.solved_t=[0 for i in range(0,self.num)]
		self.idx_users=[]
		self.a=[1 for i in range(0,self.num)]
		print("Starting Matlab")
		self.eng=matlab.engine.start_matlab()
		print("Matlab Started")

	def get_alpha(self,i,Eres,b,h,p_tx): #第i个设备
		temp_alpha=(1*Eres[i])/(0+1*b[i]*log((1+h[i]*p_tx[i]/self.N0),2))
		return sum(self.alpha[i])+temp_alpha

	def update_alpha(self,i,Eres,h,p_tx):
		solved_alpha=(1*Eres[i])/(0+1*self.solved_b[i]*log((1+h[i]*p_tx[i]/self.N0),2))
		self.alpha[i].append(solved_alpha)
		return solved_alpha

	def csv_log(self,log_name, log_var):
		with open("./log/"+self.id+"/"+self.id+"_"+log_name+".csv","a") as f:
			csvwriter=csv.writer(f)
			csvwriter.writerow(log_var)

	def get_Energy_util(self,idx,E,energy_origin,energy_threshold,e):
		if E[idx]-e[idx] >= energy_origin[idx]*energy_threshold:
			return E[idx]/e[idx]
		else:
			return 0
	
	def get_Time_util(self,idx,T,t):
		if T<t[idx]:
			return T/t[idx]
		else:
			return 1

	def choose_user(self,Eres,b,h,p_tx,Dcp,P,Loss_util,LocalAge_util,args_alpha,args_beta,args_gama,args_delta,T,energy_origin,energy_threshold): 
		#Eres是当前轮次各用户剩余能量, list;
		#b是各用户带宽, list;
		#h是信道系数, list;
		#p_tx是通信发射功率, list;
		#Dcp是计算时间, list;
		#P是计算功率, list
		#T是时间阈值, int/float
		#energy_origin 初始电量
		#energy_threshold 电量阈值
		flag=True
		e=[]
		t=[]
		for i in range(0,self.num):
			if (b[i]>0):
				e.append(int(self.H0+self.get_alpha(i,Eres,b,h,p_tx)*self.delta_H)*Dcp[i]*P[i]+self.S*p_tx[i]/(b[i]*log((1+h[i]*p_tx[i]/self.N0),2)))
			else:
				e.append(1)
		for i in range(0,self.num):
			if (b[i]>0):
				t.append(int(self.H0+self.get_alpha(i,Eres,b,h,p_tx)*self.delta_H)*Dcp[i]+self.S/(b[i]*log((1+h[i]*p_tx[i]/self.N0),2)))
			else:
				t.append(1)
		Energy_util=[self.get_Energy_util(i,Eres,energy_origin,energy_threshold,e) for i in range(0,self.num)]
		Time_util=[self.get_Time_util(i,T,t) for i in range(0,self.num)]
		Util=[(Loss_util[i]**args_alpha)*(Energy_util[i]**args_beta)*(Time_util[i]**args_gama)*(LocalAge_util[i]**args_delta) for i in range(0,self.num)]
		for i in range(0,self.num):
			if (b[i]==0):
				Util[i]=-1
		Util_temp=copy.deepcopy(Util)
		flag_cnt=0
		while flag:
			max_Util=max(Util_temp)
			chosen_user=Util_temp.index(max_Util)
			if b[chosen_user]==0:
				Util_temp[chosen_user]=-1
				continue
			if Eres[chosen_user]-e[chosen_user]<energy_origin[chosen_user]*energy_threshold:
				Util_temp[chosen_user]=-1
				flag_cnt=flag_cnt+1
			else:
				flag=False
			if flag_cnt>=self.num:
				chosen_user=-1
		if max_Util<0:
			chosen_user=-1
			print("Choose Wrong!")
		return chosen_user

	def prob_b(self,Eres,a,h,p_tx,Dcp,P,Loss_util,LocalAge_util,args_alpha,args_beta,args_gama,args_delta,T,energy_origin,energy_threshold):
		K=[self.H0+sum(self.alpha[i])*self.delta_H for i in range(0,self.num)]
		Q=[log((1+h[i]*p_tx[i]/self.N0),2) for i in range(0,self.num)]
		W=[Eres[i]/Q[i]*self.delta_H for i in range(0,self.num)]
		S=self.S
		F=[S*p_tx[i] for i in range(0,self.num)]
		D=Dcp
		P=P
		J=Loss_util
		T=T
		E=Eres
		arg1=[a[i]*1/(J[i]*T*E[i]) for i in range(0,self.num)]
		arg2=[(W[i]*D[i]+S/Q[i])*(W[i]*D[i]*P[i]+F[i]/Q[i]) for i in range(0,self.num)]
		arg3=[(W[i]*D[i]+S/Q[i])*K[i]*D[i]*P[i]+(W[i]*D[i]*P[i]+F[i]/Q[i])*K[i]*D[i] for i in range(0,self.num)]
		arg4=[K[i]*K[i]*D[i]*D[i]*P[i] for i in range(0,self.num)]
		ans=self.eng.mycvx(matlab.double(arg1),matlab.double(arg2),matlab.double(arg3),matlab.double(arg4),matlab.double([self.num]),matlab.double([self.total_B]))
		b=[ans[i][0] for i in range(0,self.num)]
		print("Solved")
		return b

	def do_optimize(self,iter,Eres,h,p_tx,Dcp,P,Loss_util,LocalAge_util,args_alpha,args_beta,args_gama,args_delta,T,energy_origin,energy_threshold):
		cnt=0
		self.solved_b=[]
		self.idx_users=[]
		self.a=[1 for i in range(0,self.num)]
		if (iter==0):
			self.idx_users=[i for i in range(0,self.num)]
			print("First Iter")
		else:
			while cnt<self.m:
				print("optimize_iter "+str(cnt))
				temp_b=self.prob_b(Eres,self.a,h,p_tx,Dcp,P,Loss_util,LocalAge_util,args_alpha,args_beta,args_gama,args_delta,T,energy_origin,energy_threshold)
				chosen_user=self.choose_user(Eres,temp_b,h,p_tx,Dcp,P,Loss_util,LocalAge_util,args_alpha,args_beta,args_gama,args_delta,T,energy_origin,energy_threshold)
				if chosen_user<0:
					return self.idx_users, 1
				self.idx_users.append(chosen_user)
				self.a[chosen_user]=0
				cnt=cnt+1
		self.a=[0 for i in range(0,self.num)]
		for i in self.idx_users:
			self.a[i]=1
		self.solved_b=self.prob_b(Eres,self.a,h,p_tx,Dcp,P,Loss_util,LocalAge_util,args_alpha,args_beta,args_gama,args_delta,T,energy_origin,energy_threshold)
		solved_alpha=[0 for i in range(self.num)]
		for i in self.idx_users:
			solved_alpha[i]=self.update_alpha(i,Eres,h,p_tx)

		e=[]
		t=[]
		for i in range(0,self.num):
			if (self.solved_b[i]>0):
				e.append(int(self.H0+self.get_alpha(i,Eres,self.solved_b,h,p_tx)*self.delta_H)*Dcp[i]*P[i]+self.S*p_tx[i]/(self.solved_b[i]*log((1+h[i]*p_tx[i]/self.N0),2)))
			else:
				e.append(1)
		for i in range(0,self.num):
			if (self.solved_b[i]>0):
				t.append(int(self.H0+self.get_alpha(i,Eres,self.solved_b,h,p_tx)*self.delta_H)*Dcp[i]+self.S/(self.solved_b[i]*log((1+h[i]*p_tx[i]/self.N0),2)))
			else:
				t.append(1)
		Energy_util=[self.get_Energy_util(i,Eres,energy_origin,energy_threshold,e) for i in range(0,self.num)]
		Time_util=[self.get_Time_util(i,T,t) for i in range(0,self.num)]
		Util=[(Loss_util[i]**args_alpha)*(Energy_util[i]**args_beta)*(Time_util[i]**args_gama)*(LocalAge_util[i]**args_delta) for i in range(0,self.num)]
		for i in range(0,self.num):
			if (self.solved_b[i]<=0):
				Util[i]=-1
				e[i]=0
				t[i]=0
		self.solved_t=copy.deepcopy(t)
		self.solved_e=copy.deepcopy(e)
		#log
		self.csv_log("Energy_util",Energy_util)
		self.csv_log("Time_util",Time_util)
		self.csv_log("Loss_util",Loss_util)
		self.csv_log("LocalAge_util",LocalAge_util)
		self.csv_log("Util",Util)
		self.csv_log("Idx_users",self.idx_users)
		self.csv_log("Local_Bandwidth",self.solved_b)
		##
		return self.idx_users, 0

	def eng_quit(self):
		self.eng.quit()
		print("Matlab Quit")

	def get_local_iter(self,i,Eres,h,p_tx):
		if self.solved_b[i]>0:
			return int(self.H0+self.get_alpha(i,Eres,self.solved_b,h,p_tx)*self.delta_H)
		return 0

	def local_energy_check(self,idx,Eres,energy_origin,energy_threshold):
		if Eres[idx]-self.solved_e[idx]<energy_origin[idx]*energy_threshold:
			return -1
		return 1

	def local_energy_update(self,idx,Eres,energy_origin,energy_threshold):
		if self.local_energy_check(idx,Eres,energy_origin,energy_threshold)>0:
			Eres[idx]=Eres[idx]-self.solved_e[idx]
			return 1
		return -1

if __name__ == "__main__":
	myid="try"
	num=10
	total_B=100
	delta_H=1
	H0=4
	S=64
	N0=1
	m=6
	Eres=[8000,8100,8200,8300,8400,8500,8600,8700,8800,8900]
	h=[3,3,3,3,3,3,3,3,3,3]
	p_tx=[2,2,2,2,2,2,2,2,2,2]
	Dcp=[11/15,11/16,11/17,11/18,11/19,11/20,11/21,11/22,11/23,11/24]
	P=[10,9,8,7,6,5,4,3,2,1]
	Loss_util=[30,30,30,30,30,30,30,30,30,30]
	LocalAge_util=[1,1,2,2,3,3,4,4,5,5]
	args_alpha=0.4
	args_beta=1
	args_gama=0.125
	args_delta=1
	T=10
	energy_origin=[8000,8100,8200,8300,8400,8500,8600,8700,8800,8900]
	energy_threshold=0.1
	optimization=optimize(myid,num,total_B,delta_H,H0,S,N0,m)
	idx_users=optimization.do_optimize(Eres,h,p_tx,Dcp,P,Loss_util,LocalAge_util,args_alpha,args_beta,args_gama,args_delta,T,energy_origin,energy_threshold)
	print(idx_users)
	optimization.eng_quit()