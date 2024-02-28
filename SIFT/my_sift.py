#coding=utf8
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import cv2
import os,sys
import scipy.ndimage
import time
import scipy
from numpy.linalg import det, lstsq, norm
from functools import cmp_to_key





###################################################### 1. 定义SIFT类 ####################################################
class  CSift:
	def __init__(self,num_octave,num_scale,sigma):
		self.sigma = sigma	#初始尺度因子
		self.num_scale = num_scale #层数
		self.num_octave = 3 #组数，后续重新计算
		self.contrast_t = 0.04#弱响应阈值
		self.eigenvalue_r = 10#hessian矩阵特征值的比值阈值
		self.scale_factor = 1.5#求取方位信息时的尺度系数
		self.radius_factor = 3#3被采样率
		self.num_bins = 36 #计算极值点方向时的方位个数
		self.peak_ratio = 0.8 #求取方位信息时，辅方向的幅度系数

###################################################### 2. 构建尺度空间 ####################################################
def pre_treat_img(img_src,sigma,sigma_camera=0.5):
	sigma_mid = np.sqrt(sigma**2 - (2*sigma_camera)**2)#因为接下来会将图像尺寸放大2倍处理，所以sigma_camera值翻倍
	img = img_src.copy()
	img = cv2.resize(img,(img.shape[1]*2,img.shape[0]*2),interpolation=cv2.INTER_LINEAR)#注意dstSize的格式，行、列对应高、宽
	img = cv2.GaussianBlur(img,(0,0),sigmaX=sigma_mid,sigmaY=sigma_mid)
	return img

def get_numOfOctave(img):
	num = round (np.log(min(img.shape[0],img.shape[1]))/np.log(2) )-1
	return num

def construct_gaussian_pyramid(img_src,sift:CSift):
	pyr=[]
	img_base = img_src.copy()
	for i in range(sift.num_octave):#共计构建octave组
		octave = construct_octave(img_base,sift.num_scale,sift.sigma) #构建每一个octave组
		pyr.append(octave)
		img_base = octave[-3]#倒数第三层的尺度与下一组的初始尺度相同，对该层进行降采样，作为下一组的图像输入
		img_base = cv2.resize(img_base,(int(img_base.shape[1]/2),int(img_base.shape[0]/2)),interpolation=cv2.INTER_NEAREST)
	return pyr

def construct_octave(img_src,s,sigma):
	octave = []
	octave.append(img_src) #输入的图像已经进行过GaussianBlur了
	k = 2**(1/s)
	for i in range(1,s+3):#为得到S层个极值结果，需要构建S+3个高斯层
		img = octave[-1].copy()
		cur_sigma = k**i*sigma
		pre_sigma = k**(i-1)*sigma
		mid_sigma = np.sqrt(cur_sigma**2 - pre_sigma**2)
		cur_img = cv2.GaussianBlur(img,(0,0),sigmaX=mid_sigma,sigmaY=mid_sigma)
		octave.append(cur_img)
	return octave
###################################################### 3. 寻找初始极值点 ##################################################
def construct_DOG(pyr):
	dog_pyr=[]
	for i in range(len(pyr)):#对于每一组高斯层
		octave = pyr[i] #获取当前组
		dog=[]
		for j in range(len(octave)-1):#对于当前层
			diff = octave[j+1]-octave[j]
			dog.append(diff)
		dog_pyr.append(dog)
	return dog_pyr

def get_keypoints(gau_pyr,dog_pyr,sift:CSift):
	key_points = []
	threshold = np.floor(0.5 * sift.contrast_t / sift.num_scale * 255) #原始图像灰度范围[0,255]
	for octave_index in range(len(dog_pyr)):#遍历每一个DoG组
		octave = dog_pyr[octave_index]#获取当前组下高斯差分层list
		for s in range(1,len(octave)-1):#遍历每一层（第1层到倒数第2层）
			bot_img,mid_img,top_img = octave[s-1],octave[s],octave[s+1] #获取3层图像数据
			board_width = 5
			x_st ,y_st= board_width,board_width
			x_ed ,y_ed = bot_img.shape[0]-board_width,bot_img.shape[1]-board_width
			for i in range(x_st,x_ed):#遍历中间层图像的所有x
				for j in range(y_st,y_ed):#遍历中间层图像的所有y
					flag = is_extreme(bot_img[i-1:i+2,j-1:j+2],mid_img[i-1:i+2,j-1:j+2],top_img[i-1:i+2,j-1:j+2],threshold)#初始判断是否为极值
					if flag:#若初始判断为极值，则尝试拟合获取精确极值位置
						reu = try_fit_extreme(octave,s,i,j,board_width,octave_index,sift)
						if reu is not None:#若插值成功，则求取方向信息，
							kp,stemp = reu
							kp_orientation =  compute_orientation(kp,octave_index,gau_pyr[octave_index][stemp],sift)
							for k in kp_orientation:#将带方向信息的关键点保存
								key_points.append(k)
	return key_points

def is_extreme(bot,mid,top,thr):
	c = mid[1][1]
	temp = np.concatenate([bot,mid,top],axis=0)
	if c>thr:
		index1 = temp>c
		flag1 = len(np.where(index1 == True)[0]) > 0
		return not flag1
	elif c<-thr:
		index2 = temp<c
		flag2 = len(np.where(index2 == True)[0]) > 0
		return not flag2
	return False

def try_fit_extreme(octave,s,i,j,board_width,octave_index,sift:CSift):
	flag = False
	# 1. 尝试拟合极值点位置
	for n in range(5):# 共计尝试5次
		bot_img, mid_img, top_img = octave[s - 1], octave[s], octave[s + 1]
		g,h,offset = fit_extreme(bot_img[i - 1:i + 2, j - 1:j + 2], mid_img[i - 1:i + 2, j - 1:j + 2],top_img[i - 1:i + 2, j - 1:j + 2])
		if(np.max(abs(offset))<0.5):#若offset的3个维度均小于0.5，则成功跳出
			flag = True
			break
		s,i,j=round(s+offset[2]),round(i+offset[1]),round(j+offset[0])#否则，更新3个维度的值，重新尝试拟合
		if i<board_width or i>bot_img.shape[0]-board_width or j<board_width or j>bot_img.shape[1]-board_width or s<1 or s>len(octave)-2:#若超出边界，直接退出
			break
	if not flag:
		return None
	# 2. 拟合成功，计算极值
	ex_value = mid_img[i,j]/255+0.5*np.dot(g, offset)#求取经插值后的极值
	if np.abs(ex_value)*sift.num_scale<sift.contrast_t: #再次进行弱响应剔除
		return None
	# 3. 消除边缘响应
	hxy=h[0:2,0:2] #获取关于x、y的hessian矩阵
	trace_h = np.trace(hxy) #求取矩阵的迹
	det_h = det(hxy) #求取矩阵的行列式
	# 若hessian矩阵的特征值满足条件（认为不是边缘）
	if det_h>0 and (trace_h**2/det_h)<((sift.eigenvalue_r+1)**2/sift.eigenvalue_r):
		kp = cv2.KeyPoint()
		kp.response = abs(ex_value)#保存响应值
		i,j = (i+offset[1]),(j+offset[0])#更新精确x、y位置
		kp.pt =  j/bot_img.shape[1],i/bot_img.shape[0] #这里保存坐标的百分比位置，免去后续在不同octave上的转换
		kp.size = sift.sigma*(2**( (s+offset[2])/sift.num_scale) )* 2**(octave_index)# 保存sigma(o,s)
		kp.octave = octave_index + s * (2 ** 8) + int(round((offset[2] + 0.5) * 255)) * (2 ** 16)# 低8位存放octave的index，中8位存放s整数部分，剩下的高位部分存放s的小数部分
		return kp,s
	return None

def fit_extreme(bot,mid,top):#插值求极值
	arr = np.array([bot,mid,top])/255
	g = get_gradient(arr)
	h = get_hessian(arr)
	rt =   -lstsq(h, g, rcond=None)[0]#求解方程组
	return g,h,rt

def get_gradient(arr): #获取一阶梯度
	dx = (arr[1,1,2]-arr[1,1,0])/2
	dy = (arr[1,2,1] - arr[1,0,1])/2
	ds = (arr[2,1,1] - arr[0,1,1])/2
	return np.array([dx, dy, ds])
def get_hessian(arr): #获取三维hessian矩阵
	dxx = arr[1,1,2]-2*arr[1,1,1] + arr[1,1,0]
	dyy = arr[1,2,1]-2*arr[1,1,1] + arr[1,0,1]
	dss = arr[2,1,1]-2*arr[1,1,1] + arr[0,1,1]
	dxy = 0.25*( arr[1,0,0]+arr[1,2,2]-arr[1,0,2] - arr[1,2,0]  )
	dxs = 0.25*( arr[0,1,0]+arr[2,1,2] -arr[0,1,2] - arr[2,1,0])
	dys = 0.25*( arr[0,0,1]+arr[2,2,1]- arr[0,2,1] -arr[2,0,1])
	return np.array([[dxx,dxy,dxs],[dxy,dyy,dys],[dxs,dys,dss]])
###################################################### 4. 计算方位信息  ##################################################
def compute_orientation(kp,octave_index,img,sift:CSift):
	keypoints_with_orientations = []
	cur_scale = kp.size / (2**(octave_index)) #除去组信息o，不知为何？莫非因为输入图像img已经是进行了降采样的图像，涵盖了o的信息？
	radius = round(sift.radius_factor*sift.scale_factor*cur_scale)#求取邻域半径
	weight_sigma = -0.5 / ((sift.scale_factor*cur_scale) ** 2)#高斯加权运算系数
	raw_histogram = np.zeros(sift.num_bins)#初始化方位数组
	cx = round( kp.pt[0]*img.shape[1] )#获取极值点位置x
	cy = round( kp.pt[1]*img.shape[0] )#获取极值点位置y
	# 1.计算邻域内所有点的梯度值、梯度角，并依据梯度角将梯度值分配到相应的方向组中
	for y in range(cy-radius, cy+radius + 1): # 高，对应行
		for x in range(cx-radius, cx+radius + 1):# 宽，对应列
			if y > 0 and y < img.shape[0] - 1 and x > 0 and x < img.shape[1] - 1 :
				dx = img[y, x + 1] - img[y, x - 1]
				dy = img[y - 1, x] - img[y + 1, x]
				mag = np.sqrt(dx ** 2 + dy ** 2)#计算梯度模
				angle = np.rad2deg(np.arctan2(dy, dx))#计算梯度角
				if angle < 0:
					angle = angle + 360
				angle_index = round(angle / (360 / sift.num_bins))
				angle_index = angle_index % sift.num_bins
				weight = np.exp(weight_sigma * ((y-cy)**2 +(x-cx)**2 ))#根据x、y离中心点的位置，计算权重
				raw_histogram[angle_index] = raw_histogram[angle_index] + mag * weight#将模值分配到相应的方位组上
	# 2. 对方向组直方图进行平滑滤波
	h = raw_histogram
	ha2 = np.append(h[2:],(h[0],h[1])) # np.roll will be better
	hm2 = np.append((h[-2],h[-1]),h[:-2])
	ha1 = np.append(h[1:], h[0])
	hm1 = np.append(h[-1], h[:-1])
	smooth_histogram = ( ha2+hm2 + 4*(ha1+hm1) + 6*h)/16
	# 3. 计算极值点的主方向和辅方向
	s = smooth_histogram
	max_v = max(s)# 找最大值
	s1 = np.roll(s,1)
	s2 = np.roll(s,-1)
	index1 = s>=s1
	index2 = s>=s2
	index = np.where( np.logical_and(index1,index2)==True )[0] #找到所有极值点位置
	for i in index:
		peak_v = s[i]
		if peak_v >= sift.peak_ratio * max_v: #若大于阈值，则保留，作为主/辅方向
			left_v = s[(i-1)%sift.num_bins]
			right_v = s[(i+1)%sift.num_bins]
			index_fit= ( i+0.5*(left_v-right_v)/(left_v+right_v-2*peak_v) )%sift.num_bins#插值得到精确极值位置
			angle = 360-index_fit/sift.num_bins*360 #计算精确的方位角
			new_kp =cv2.KeyPoint(*kp.pt, kp.size, angle, kp.response, kp.octave) #在关键点中，加入方向信息
			keypoints_with_orientations.append(new_kp)
	return keypoints_with_orientations
#################################################### 5. 计算关键点的特征描述符 #############################################
def get_descriptor(kps,gau_pyr,win_N=4, num_bins=8, scale_multiplier=3, des_max_value=0.2):
	descriptors = []
	for kp in kps:
		octave, layer  =kp.octave & 255 , (kp.octave >> 8)&255
		image = gau_pyr[octave][layer]
		img_rows,img_cols = image.shape
		bins_per_degree = num_bins / 360.
		angle = 360. - kp.angle #旋转角度为关键点方向角的负数
		cos_angle = np.cos(np.deg2rad(angle))
		sin_angle = np.sin(np.deg2rad(angle))
		weight_multiplier = -0.5 / ((0.5 * win_N) ** 2)#高斯加权运算，方差为0.5×d
		row_bin_list = []#存放每个邻域点对应4×4个小窗口中的哪一个（行）
		col_bin_list = []#存放每个邻域点对应4×4个小窗口中的哪一个（列）
		magnitude_list = []#存放每个邻域点的梯度幅值
		orientation_bin_list = []#存放每个邻域点的梯度方向角所处的方向组
		histogram_tensor = np.zeros((win_N + 2, win_N + 2, num_bins))#存放4×4×8个描述符，但为防止计算时边界溢出，在行、列的首尾各扩展一次

		hist_width = scale_multiplier * kp.size/(2**(octave)) # 3×sigma，每个小窗口的边长
		radius = int(round(hist_width * np.sqrt(2) * (win_N + 1) * 0.5))
		radius = int(min(radius, np.sqrt(img_rows ** 2 + img_cols ** 2)))

		for row in range(-radius, radius + 1):
			for col in range(-radius, radius + 1):
				row_rot = col * sin_angle + row * cos_angle#计算旋转后的坐标
				col_rot = col * cos_angle - row * sin_angle#计算旋转后的坐标
				row_bin = (row_rot / hist_width) + 0.5 * win_N - 0.5 #对应4×4子区域的下标（行）
				col_bin = (col_rot / hist_width) + 0.5 * win_N - 0.5#对应在4×4子区域的下标（列）
				if row_bin > -1 and row_bin < win_N and col_bin > -1 and col_bin < win_N:#邻域的点在旋转后，仍然处于4×4的区域内，
					window_row = int(round(kp.pt[1]*image.shape[0] + row))#计算对应原图的row
					window_col = int(round(kp.pt[0]*image.shape[1] + col))#计算对应原图的col
					if window_row > 0 and window_row < img_rows - 1 and window_col > 0 and window_col < img_cols - 1:
						dx = image[window_row, window_col + 1] - image[window_row, window_col - 1]#直接在旋转前的图上计算梯度，因为旋转时，都旋转了，不影响大小
						dy = image[window_row - 1, window_col] - image[window_row + 1, window_col]#直接在旋转前的图上计算梯度，因为旋转时，都旋转了，不影响大小
						gradient_magnitude = np.sqrt(dx * dx + dy * dy)
						gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
						weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))#不明白为什么要处以小窗口的边长，是要以边长为单位？
						row_bin_list.append(row_bin)
						col_bin_list.append(col_bin)
						magnitude_list.append(weight * gradient_magnitude)
						orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)#因为梯度角是旋转前的，所以还要叠加上旋转的角度

		#将magnitude分配到4*4*8（d*d*num_bins）的各区域中，即分配到histogram_tensor数组中
		for r,c,o,m in zip(row_bin_list,col_bin_list,orientation_bin_list,magnitude_list):
			ri,ci,oi = np.floor([r,c,o]).astype(int)
			rf,cf,of = [r,c,o]-np.array([ri,ci,oi])	#rf越大，越偏离当前行，同理cf，of
			#先按行分解
			c0 = m*(1-rf)#当前行的分量
			c1 = m*rf #下一行的分量
			#对每一个行分量，按列分解
			c00 = c0*(1-cf)#当前行、当前列
			c01 = c0*cf##当前行、下一列
			c10 = c1*(1-cf)#下一行、当前列
			c11=c1*cf#下一行、下一列
			#对每一个行+列分量，按方向角分解
			c000, c001 = c00*(1-of),c00*of
			c010, c011= c01*(1-of),c01*of
			c100,c101 = c10*(1-of), c10*of
			c110,c111 = c11*(1-of),c11*of
			# 数值填入到数组中
			histogram_tensor[ri+1,ci+1,oi] += c000
			histogram_tensor[ri + 1, ci + 1, (oi+1)%num_bins] += c001
			histogram_tensor[ri + 1, ci + 2, oi] += c010
			histogram_tensor[ri + 1, ci + 2, (oi + 1) % num_bins] += c011
			histogram_tensor[ri + 2, ci + 1, oi] += c100
			histogram_tensor[ri + 2, ci + 1, (oi + 1) % num_bins] += c101
			histogram_tensor[ri + 2, ci + 2, oi] += c110
			histogram_tensor[ri + 2, ci + 2, (oi + 1) % num_bins] += c111

		des_vec = histogram_tensor[1:-1,1:-1,:].flatten()#转成一维向量形式
		#des_vec[des_vec > des_max_value*np.linalg.norm(des_vec)] = des_max_value*np.linalg.norm(des_vec)
		#des_vec = des_vec / np.linalg.norm(des_vec)
		des_vec = des_vec/np.linalg.norm(des_vec)
		des_vec[des_vec>des_max_value] = des_max_value
		des_vec = np.round(512*des_vec)
		des_vec[des_vec<0]=0
		des_vec[des_vec>255]=255
		descriptors.append(des_vec)
	return descriptors


def sort_method(kp1:cv2.KeyPoint,kp2:cv2.KeyPoint):
	if kp1.pt[0] != kp2.pt[0]:
		return kp1.pt[0] - kp2.pt[0]
	if kp1.pt[1] != kp2.pt[1]:
		return kp1.pt[1] - kp2.pt[1]
	if kp1.size != kp2.size:
		return kp1.size - kp2.size
	if kp1.angle != kp2.angle:
		return kp1.angle - kp2.angle
	if kp1.response != kp2.response:
		return kp1.response - kp2.response
	if kp1.octave != kp2.octave:
		return kp1.octave - kp2.octave
	return kp1.class_id - kp2.class_id

def remove_duplicate_points(keypoints):
	keypoints.sort(key=cmp_to_key(sort_method))
	unique_keypoints = [keypoints[0]]

	for next_keypoint in keypoints[1:]:
		last_unique_keypoint = unique_keypoints[-1]
		if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
				last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
				last_unique_keypoint.size != next_keypoint.size or \
				last_unique_keypoint.angle != next_keypoint.angle:
			unique_keypoints.append(next_keypoint)
	return unique_keypoints

#################################################### 6. 匹配 ############################################################
def do_match(img_src1,kp1,des1,img_src2,kp2,des2,embed=1,pt_flag=0,MIN_MATCH_COUNT = 10):
	## 1. 对关键点进行匹配 ##
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	des1, des2 = np.array(des1).astype(np.float32), np.array(des2).astype(np.float32)#需要转成array
	matches = flann.knnMatch(des1, des2, k=2)  # matches为list，每个list元素由2个DMatch类型变量组成,分别是最邻近和次邻近点

	good_match = []
	for m in matches:
		if m[0].distance < 0.7 * m[1].distance:  # 如果最邻近和次邻近的距离差距较大,则认可
			good_match.append(m[0])
	## 2. 将2张图画在同一张图上 ##
	img1 = img_src1.copy()
	img2 = img_src2.copy()
	h1, w1 = img1.shape[0],img1.shape[1]
	h2, w2 = img2.shape[0],img2.shape[1]
	new_w = w1 + w2
	new_h = np.max([h1, h2])
	new_img =  np.zeros((new_h, new_w,3), np.uint8) if len(img_src1.shape)==3 else  np.zeros((new_h, new_w), np.uint8)
	h_offset1 = int(0.5 * (new_h - h1))
	h_offset2 = int(0.5 * (new_h - h2))
	if len(img_src1.shape) == 3:
		new_img[h_offset1:h_offset1 + h1, :w1,:] = img1  # 左边画img1
		new_img[h_offset2:h_offset2 + h2, w1:w1 + w2,:] = img2  # 右边画img2
	else:
		new_img[h_offset1:h_offset1 + h1, :w1] = img1  # 左边画img1
		new_img[h_offset2:h_offset2 + h2, w1:w1 + w2] = img2  # 右边画img2
	##3. 两幅图存在足够的匹配点，两幅图匹配成功，将匹配成功的关键点进行连线 ##
	if len(good_match) > MIN_MATCH_COUNT:
		src_pts = []
		dst_pts = []
		mag_err_arr=[]
		angle_err_arr=[]
		for m in good_match:
			if pt_flag==0:#point是百分比
				src_pts.append([kp1[m.queryIdx].pt[0] * img1.shape[1], kp1[m.queryIdx].pt[1] * img1.shape[0]])#保存匹配成功的原图关键点位置
				dst_pts.append([kp2[m.trainIdx].pt[0] * img2.shape[1], kp2[m.trainIdx].pt[1] * img2.shape[0]])#保存匹配成功的目标图关键点位置
			else:
				src_pts.append([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]])  # 保存匹配成功的原图关键点位置
				dst_pts.append([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]])  # 保存匹配成功的目标图关键点位置

			mag_err = np.abs(kp1[m.queryIdx].response - kp2[m.trainIdx].response) / np.abs(kp1[m.queryIdx].response )
			angle_err = np.abs(kp1[m.queryIdx].angle - kp2[m.trainIdx].angle)
			mag_err_arr.append(mag_err)
			angle_err_arr.append(angle_err)

		if embed!=0 :#若图像2是图像1内嵌入另一个大的背景中，则在图像2中，突出显示图像1的边界
			M = cv2.findHomography(np.array(src_pts), np.array(dst_pts), cv2.RANSAC, 5.0)[0]  # 根据src和dst关键点，寻求变换矩阵
			src_w, src_h = img1.shape[1], img1.shape[0]
			src_rect = np.array([[0, 0], [src_w - 1, 0], [src_w - 1, src_h - 1], [0, src_h - 1]]).reshape(-1, 1, 2).astype(
				np.float32)  # 原始图像的边界框
			dst_rect = cv2.perspectiveTransform(src_rect, M)  # 经映射后，得到dst的边界框
			img2 = cv2.polylines(img2, [np.int32(dst_rect)], True, 255, 3, cv2.LINE_AA)  # 将边界框画在dst图像上，突出显示
			if len(new_img.shape) == 3:
				new_img[h_offset2:h_offset2 + h2, w1:w1 + w2,:] = img2  # 右边画img2
			else:
				new_img[h_offset2:h_offset2 + h2, w1:w1 + w2] = img2  # 右边画img2

		new_img = new_img if len(new_img.shape) == 3 else  cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR)
		# 连线
		for pt1, pt2 in zip(src_pts, dst_pts):
			cv2.line(new_img, tuple(np.int32(np.array(pt1) + [0, h_offset1])),
					 tuple(np.int32(np.array(pt2) + [w1, h_offset2])), color=(0, 0, 255))
	return new_img

#################################################### 7. 测试移动和旋转 ####################################################
def move_rotate_sift_test():
	img_src = cv2.imread('box.png', 0)
	img_src1 = img_src.copy()
	#### 情形1.旋转90度 ###
	img_src2 = img_src1.transpose()
	img_src2 = np.fliplr(img_src2)
	#### 情形2. 景色移出视野 ###
	img_src2 = img_src1[:,50:]
	#### 情形3. 放射变换 ###
	points1 = np.float32([[81, 30], [378, 80], [13, 425]])
	points2 = np.float32([[0, 0], [300, 0], [100, 300]])
	affine_matrix = cv2.getAffineTransform(points1, points2)

	img_src2 = cv2.warpAffine(img_src1, affine_matrix, (0, 0), flags=cv2.INTER_CUBIC,
							  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

	####　做sift ####
	sift.num_octave = get_numOfOctave(img_src1)
	opencv_sift = cv2.SIFT.create(nfeatures=None, nOctaveLayers=sift.num_octave,
								  contrastThreshold=sift.contrast_t, edgeThreshold=sift.eigenvalue_r, sigma=sift.sigma)
	kp1 = opencv_sift.detect(img_src1)
	kp1, des1 = opencv_sift.compute(img_src1, kp1)

	sift.num_octave = get_numOfOctave(img_src2)
	opencv_sift = cv2.SIFT.create(nfeatures=None, nOctaveLayers=sift.num_octave,
								  contrastThreshold=sift.contrast_t, edgeThreshold=sift.eigenvalue_r, sigma=sift.sigma)
	kp2 = opencv_sift.detect(img_src2)
	kp2, des2 = opencv_sift.compute(img_src2, kp2)

	reu_img = do_match_compare(img_src1, kp1, des1, img_src2, kp2, des2, embed=0, pt_flag=1, MIN_MATCH_COUNT=3)
	cv2.imshow('reu', reu_img)


	return

def do_match_compare(img_src1,kp1,des1,img_src2,kp2,des2,embed=1,pt_flag=0,MIN_MATCH_COUNT = 10):
	## 1. 对关键点进行匹配 ##
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	des1, des2 = np.array(des1).astype(np.float32), np.array(des2).astype(np.float32)#需要转成array
	matches = flann.knnMatch(des1, des2, k=2)  # matches为list，每个list元素由2个DMatch类型变量组成,分别是最邻近和次邻近点

	good_match = []
	for m in matches:
		if m[0].distance < 0.5 * m[1].distance:  # 如果最邻近和次邻近的距离差距较大,则认可
			good_match.append(m[0])
	## 2. 将2张图画在同一张图上 ##
	img1 = img_src1.copy()
	img2 = img_src2.copy()
	h1, w1 = img1.shape[0],img1.shape[1]
	h2, w2 = img2.shape[0],img2.shape[1]
	new_w = w1 + w2
	new_h = np.max([h1, h2])
	new_img = np.zeros((new_h, new_w), np.uint8)
	h_offset1 = int(0.5 * (new_h - h1))
	h_offset2 = int(0.5 * (new_h - h2))
	new_img[h_offset1:h_offset1 + h1, :w1] = img1  # 左边画img1
	new_img[h_offset2:h_offset2 + h2, w1:w1 + w2] = img2  # 右边画img2
	##3. 两幅图存在足够的匹配点，两幅图匹配成功，将匹配成功的关键点进行连线 ##
	if len(good_match) > MIN_MATCH_COUNT:
		src_pts = []
		dst_pts = []
		mag_err_arr=[] #保存匹配的关键点，在梯度幅值上的相对误差
		angle_err_arr=[]#保存匹配的关键点，在梯度方向上的绝对误差
		for m in good_match:
			if pt_flag==0:#point是百分比
				src_pts.append([kp1[m.queryIdx].pt[0] * img1.shape[1], kp1[m.queryIdx].pt[1] * img1.shape[0]])#保存匹配成功的原图关键点位置
				dst_pts.append([kp2[m.trainIdx].pt[0] * img2.shape[1], kp2[m.trainIdx].pt[1] * img2.shape[0]])#保存匹配成功的目标图关键点位置
			else:
				src_pts.append([kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]])  # 保存匹配成功的原图关键点位置
				dst_pts.append([kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]])  # 保存匹配成功的目标图关键点位置

			mag_err = np.abs(kp1[m.queryIdx].response - kp2[m.trainIdx].response) / np.abs(kp1[m.queryIdx].response ) *100
			angle_err = (kp1[m.queryIdx].angle - kp2[m.trainIdx].angle)%360
			mag_err_arr.append(mag_err)
			angle_err_arr.append(angle_err)
		new_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR)
		plt.figure()
		plt.title('mag_err (%)')
		plt.plot(mag_err_arr)

		plt.figure()
		plt.title('angle_err (degree)')
		plt.plot(angle_err_arr)
		plt.show()
		# 连线
		index = np.argsort(mag_err_arr)#进行有小到大排序

		for i in range(0,5): #画出误差最小的5个匹配点连线
			pt1, pt2 = src_pts[index[i]], dst_pts[index[i]]
			cv2.line(new_img, tuple(np.int32(np.array(pt1) + [0, h_offset1])),
					 tuple(np.int32(np.array(pt2) + [w1, h_offset2])), color=(0, 0, 255))
		for i in range(-3, 0):#画出误差最大的3个匹配点连线
			pt1, pt2 = src_pts[index[i]], dst_pts[index[i]]
			cv2.line(new_img, tuple(np.int32(np.array(pt1) + [0, h_offset1])), tuple(np.int32(np.array(pt2) + [w1, h_offset2])), color=(255, 0, 0))
	return new_img



def do_sift(img_src,sift:CSift):
	img = img_src.copy().astype(np.float32)
	img = pre_treat_img(img,sift.sigma)
	sift.num_octave = get_numOfOctave(img)
	gaussian_pyr = construct_gaussian_pyramid(img,sift)
	dog_pyr = construct_DOG(gaussian_pyr)
	key_points = get_keypoints(gaussian_pyr,dog_pyr,sift)
	key_points = remove_duplicate_points(key_points)
	descriptor = get_descriptor(key_points,gaussian_pyr)
	return key_points,descriptor



if __name__ == '__main__':
	MIN_MATCH_COUNT = 10
	sift = CSift(num_octave=4,num_scale=3,sigma=1.6)
	img_src1 = cv2.imread('box.png',-1)
	img_src2 = cv2.imread('box_in_scene.png', -1)

	img_src1 = cv2.imread('img_day.png', -1)
	img_src2 = cv2.imread('img_night.png', -1)
	# (512, 512, 4)变为(512,512)
	img_src1 = img_src1[:, :, :1]
	img_src2 = img_src2[:, :, :1]


	print(img_src1.shape)
	print(img_src2.shape)


	# img_src2 = cv2.resize(img_src2, (0, 0), fx=.5, fy=.5)
	# img_src1 = cv2.resize(img_src1, (0, 0), fx=.25, fy=.25)
	# '''
	# # 1. 使用本sift算子
	# kp1, des1 = do_sift(img_src1, sift)
	# kp2, des2 = do_sift(img_src2, sift)
	# pt_flag = 0
	# '''
	# # 2. 或者使用opencv自带sift算子
	# sift.num_octave = get_numOfOctave(img_src1)
	# opencv_sift = cv2.SIFT.create(nfeatures=None, nOctaveLayers=sift.num_octave,
	# 							  contrastThreshold=sift.contrast_t, edgeThreshold=sift.eigenvalue_r, sigma=sift.sigma)
	# kp1 = opencv_sift.detect(img_src1)
	# kp1,des1 = opencv_sift.compute(img_src1,kp1)

	# sift.num_octave = get_numOfOctave(img_src2)
	# opencv_sift = cv2.SIFT.create(nfeatures=None, nOctaveLayers=sift.num_octave,
	# 							  contrastThreshold=sift.contrast_t, edgeThreshold=sift.eigenvalue_r, sigma=sift.sigma)
	# kp2 = opencv_sift.detect(img_src2)
	# kp2, des2 = opencv_sift.compute(img_src2, kp2)
	# pt_flag = 1
	# 使用本sift算子
	kp1, des1 = do_sift(img_src1, sift)
	kp2, des2 = do_sift(img_src2, sift)	
	pt_flag = 0



	# 3. 做匹配
	reu_img = do_match(img_src1, kp1, des1, img_src2, kp2, des2, embed=1, pt_flag=pt_flag,MIN_MATCH_COUNT=3)
	plt.figure()
	plt.imshow(reu_img)
	plt.show()

	# cv2.imshow('reu',reu_img)
	# cv2.imwrite('reu.tif',reu_img)






	print('emd')



def GaussianBlur_test(img,sigma):
	img1 = img.copy()
	img2 = img.copy()
	k = 2**(1/3)
	sigma1,sigma2 = k*sigma,k*k*sigma
	mid_sigma = np.sqrt(sigma2 ** 2 - sigma1 ** 2)
	reu1 = cv2.GaussianBlur(img1,(0,0),sigmaX=sigma1,sigmaY=sigma1)
	reu1 = cv2.GaussianBlur(reu1,(0,0),sigmaX=mid_sigma,sigmaY=mid_sigma)
	reu2 = cv2.GaussianBlur(img2,(0,0),sigmaX=sigma2,sigmaY=sigma2)
	err = reu1 - reu2
	print(np.max(abs(err) ) )
	return

