# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:16:04 2017
@author: Yusen Ye (ccyeyusen@sina.com)
@brief: MSTD is a generic and efficient method to  identify multi-scale topological domains (MSTD) 
        from symmetric Hi-C and other high resolution asymmetric promoter capture Hi-C datasets
@version 0.0.1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colormap import Color, Colormap

def _domain_only_diagonal(Data,win_n,distance):
    Dsize=Data.shape[0]
    #step1.1
    pdensity=np.zeros(Dsize)
    DEN_Dict={}
    for ip in range(Dsize):
        begin_i=ip-win_n+1
        if begin_i<0:
            begin_i=0
        end_i=ip+win_n-1
        if end_i>Dsize-1:
            end_i=Dsize-1     
        pdensity[ip]=np.mean(Data[begin_i:ip+1,:][:,ip:end_i+1])
        DEN_Dict[ip]=pdensity[ip]
    
    #step1.2
    Max_step=100
    NDP_Dict={}
    ASS_Dict={}
    for ip in np.arange(0,Dsize):
        for step in np.arange(0,max(ip,Dsize-ip)):
            if ip-step>=0:
                up_point=pdensity[ip-step]
                if up_point>pdensity[ip]:
                    ASS_Dict[ip]=ip-step
                    break
            if ip+step<=Dsize-1:
                down_point=pdensity[ip+step]
                if down_point>pdensity[ip]:
                    ASS_Dict[ip]=ip+step
                    break
            if step>Max_step:
                ASS_Dict[ip]=ip
                break
        NDP_Dict[ip]=step
    Thr_den=np.percentile(pdensity,30)
    point_assign={}
    for temp in DEN_Dict:
        point_assign[temp]=0
    class_num=1
    join_num=0
    centers=[]
    for item in DEN_Dict:
        den=DEN_Dict[item]
        dist=NDP_Dict[item]
        if (den>Thr_den and dist>distance):
            point_assign[item]=class_num
            class_num=class_num+1
            join_num=join_num+1
            centers.append(item)
            ASS_Dict[item]=item
    
    old_join_num=0
    new_join_num=join_num
    while old_join_num!=new_join_num:
        old_join_num=join_num
        for item in DEN_Dict:
            #if ((NDP_Dict[item]<distance)or(DEN_Dict[item]>Thr_den)):
                if ASS_Dict[item]==item:
                    continue
                if point_assign[ASS_Dict[item]]!=0:
                    if point_assign[item]==0:
                        point_assign[item]=point_assign[ASS_Dict[item]]
                        join_num=join_num+1
        new_join_num=join_num
    
    for item in DEN_Dict:
        if point_assign[item]!=0:
            temp=centers[int(point_assign[item])-1]
            if DEN_Dict[item]<DEN_Dict[temp]/3 :
                #print item
                point_assign[item]=0
    NLP_Dict={}
    for ip in np.arange(0,Dsize):
        if point_assign[ip]==0:
            NLP_Dict[ip]=0
            continue
        for step in np.arange(0,max(ip,Dsize-ip)):
            if ip-step>=0:
                up_point=pdensity[ip-step]
                if up_point<pdensity[ip]:
                    break
            if ip+step<=Dsize-1:
                down_point=pdensity[ip+step]
                if down_point<pdensity[ip]:
                    break
            if step>Max_step:
                break
        NLP_Dict[ip]=step
    Thr_den1=np.percentile(pdensity,70)
    corrbound=[]
    for item in DEN_Dict:
        den=DEN_Dict[item]
        dist=NLP_Dict[item]
        if (den<Thr_den1 and dist>=distance):
            corrbound.append(item)
    Dict={'density':DEN_Dict,'distance':NDP_Dict,'ass_point':ASS_Dict,'point_assign':point_assign}
    DF=pd.DataFrame(Dict)
    return DF,centers,corrbound,class_num

def _generate_density_con(Data,win,thr):
    Dsize=Data.shape
    if Dsize[0]==Dsize[1]:
        Dsize=Data.shape
        M_density=np.zeros(Dsize)
        DEN_Dict={}
        for i in range(Dsize[0]):
            for j in range(Dsize[1]):
                if Data[i,j]>thr or i==j:
                    begin_i=i-win[0]
                    if begin_i<0:
                        begin_i=0
                    begin_j=j-win[1]
                    if begin_j<0:
                        begin_j=0
                    
                    end_i=i+win[0]
                    if end_i>Dsize[0]-1:
                        end_i=Dsize[0]-1
                    
                    end_j=j+win[1]
                    if end_j>Dsize[1]-1:
                        end_j=Dsize[1]-1
                    M_density[i,j]=np.mean(Data[begin_i:end_i,begin_j:end_j])+np.random.random(1)/1000.0                
                    DEN_Dict[(i,j)]=M_density[i,j]
          
    else:
        M_density=np.zeros(Dsize)
        DEN_Dict={}
        for i in range(Dsize[0]):
            for j in range(Dsize[1]):
                if Data[i,j]>thr:
                    begin_i=i-win[0]
                    if begin_i<0:
                        begin_i=0
                    begin_j=j-win[1]
                    if begin_j<0:
                        begin_j=0
                    
                    end_i=i+win[0]
                    if end_i>Dsize[0]-1:
                        end_i=Dsize[0]-1
                    
                    end_j=j+win[1]
                    if end_j>Dsize[1]-1:
                        end_j=Dsize[1]-1
                    M_density[i,j]=np.mean(Data[begin_i:end_i,begin_j:end_j])+np.random.random(1)/1000.0                
                    DEN_Dict[(i,j)]=M_density[i,j]
                    #print M_density[i,j]
    return M_density, DEN_Dict
def _find_highpoints_v2(DEN_Dict,ratio=1):
    Dis=50
    NDP_Dict={}
    ASS_Dict={}
    for item in DEN_Dict: 
        #item=ASS_Dict[item]; item
        NDP_Dict[item]=np.linalg.norm((Dis,Dis*ratio))
        ASS_Dict[item]=item
        for step in np.arange(1,Dis+1,1):
            step_point=[(item[0]+st,item[1]+ra) for st in np.arange(-step,step+1) for ra in np.arange(-step*ratio,step*ratio+1) if (abs(st)==step or ratio*(step-1)<abs(ra)<=ratio*step) ]
            step_point=[point for point in step_point if point in DEN_Dict]
            distance_index=[(np.linalg.norm(((item[0]-temp[0])*ratio,item[1]-temp[1])),temp) for temp in step_point if DEN_Dict[temp]>DEN_Dict[item]]
            distance_index.sort()
            for ind in distance_index:
                if DEN_Dict[ind[1]]>DEN_Dict[item]:
                    NDP_Dict[item]=ind[0]
                    ASS_Dict[item]=ind[1]
                    break
            if len(distance_index)>0:
                break
    return NDP_Dict, ASS_Dict
  
def _assign_class(DEN_Dict,NDP_Dict,ASS_Dict,Thr_den,Thr_dis):
    point_assign={}
    for temp in DEN_Dict:
        point_assign[temp]=0    
    class_num=1
    join_num=0
    centers=[]
    for item in DEN_Dict:
        den=DEN_Dict[item]
        dist=NDP_Dict[item]
        #value=den*dist      
        if (den>Thr_den) and (dist>Thr_dis):
            point_assign[item]=class_num
            class_num=class_num+1
            join_num=join_num+1
            centers.append(item)
            ASS_Dict[item]=item
    
    Al=len(DEN_Dict)
    old_join_num=0
    new_join_num=join_num
    while old_join_num!=new_join_num:
        old_join_num=join_num
        for item in DEN_Dict:
            if ((NDP_Dict[item]<Thr_dis)or(DEN_Dict[item]>Thr_den)) :
                if ASS_Dict[item]==item:
                    continue
                if point_assign[ASS_Dict[item]]!=0:
                    if point_assign[item]==0:
                        point_assign[item]=point_assign[ASS_Dict[item]]
                        join_num=join_num+1
        new_join_num=join_num
    return point_assign,class_num-1,centers

def _get_region_den2(DEN_Dict,NDP_Dict,ASS_Dict,point_assign,win,centers):  
    Thr=np.percentile(pd.Series(DEN_Dict),10)
    for item in DEN_Dict:
        if point_assign[item]!=0:
            temp=centers[point_assign[item]-1]
            #if DEN_Dict[item]<np.min(DEN_Dict[temp]/3,Thr):
            if DEN_Dict[item]<Thr:
                point_assign[item]=0
    Dict={'density':DEN_Dict,'distance':NDP_Dict,'ass_point':ASS_Dict,'point_assign':point_assign}
    DF=pd.DataFrame(Dict)         
    return DF   



def _get_region_piont2(matrix_data,DF,win,centers):
    bound=np.zeros((len(centers),4))
    for i, cen in enumerate(centers):
        cen_index=DF[DF["point_assign"]==i+1].index
        indictor=np.zeros(matrix_data.shape)
        row=[]
        col=[]
        for item in cen_index:
            row.append(item[0])
            col.append(item[1])
            indictor[item[0],item[1]]=1
        
        M_temp=matrix_data*indictor
        
        left=np.min(np.array(row))
        right=np.max(np.array(row))
        bottom=np.min(np.array(col))
        upper=np.max(np.array(col))
        
        #print left,right,bottom,upper
        #MM=matrix_data[left:right+1,bottom:upper+1]
        #plt.subplots(figsize=(8,8))
        #sns.heatmap(MM[:,::-1])
        #初始化
        corr=0.99
        #left
        M_left=M_temp[left:cen[0]+1,bottom:upper+1]
        M_left=M_left[::-1,:]
        sum_left=np.sum(M_left)
        #right
        M_right=M_temp[cen[0]:right+1,bottom:upper+1]
        sum_right=np.sum(M_right)
        #bottom
        M_bottom=M_temp[left:right+1,bottom:cen[1]+1]
        M_bottom=M_bottom[:,::-1]
        sum_bottom=np.sum(M_bottom)
        #upper
        M_upper=M_temp[left:right+1,cen[1]:upper+1]
        sum_upper=np.sum(M_upper)
        
        #left
        sum_temp=0
        for step in range(cen[0]-left+1):
            sum_temp=sum_temp+np.sum(M_left[step,:])
            if sum_left==0:
                left=cen[0]-step
                break
            #print sum_temp, sum_left
            #print sum_temp/sum_left
            if sum_temp/sum_left>corr:
                left=cen[0]-step+1
                #print left
                break
        
        #right
        sum_temp=0
        for step in range(right+1-cen[0]):
            sum_temp=sum_temp+np.sum(M_right[step,:])
            if sum_right==0:
                right=cen[0]+step
                break
            if sum_temp/sum_right>corr:
                right=cen[0]+step+1
                #print sum_temp/sum_right,right
                break
        
        #bottom
        sum_temp=0
        for step in range(cen[1]-bottom+1):
            sum_temp=sum_temp+np.sum(M_bottom[:,step])
            if sum_bottom==0:
                bottom=cen[1]-step
                break
            if sum_temp/sum_bottom>corr:
                bottom=cen[1]-step+1
                #print sum_temp/sum_bottom,bottom
                break
        #upper
        sum_temp=0
        for step in range(upper+1-cen[1]):
            sum_temp=sum_temp+np.sum(M_upper[:,step])
            if sum_upper==0:
                upper=cen[1]+step
                break
            if sum_temp/sum_upper>corr:
                upper=cen[1]+step+1
                break
        bound[i,:]=np.array([upper,bottom,left,right])
    
    Bound=pd.DataFrame(bound,columns=['upper','bottom','left','right'])
    
    Centers=pd.DataFrame(centers,columns=['cen_x','cen_y'])
    
    Results=pd.concat([Bound,Centers],axis=1)    
    
    Results=Results.loc[(Results['upper']!=Results['bottom'])*(Results['left']!=Results['right'])]
    
    return Results





def _def_strmouOnCHiC(matrix_data,win_n,thr_dis=10):
    #matrix size
    Mat_size=matrix_data.shape
    print ("Matrix size:"+str(Mat_size[0])+'*'+str(Mat_size[1]) )  
    ratio=matrix_data.shape[1]//matrix_data.shape[0]
    #computing density threshold
    point_num=thr_dis*matrix_data.shape[0]/6
    #print "Effective points:"+str(point_num)
    percent=1-point_num/float(matrix_data.shape[0]*matrix_data.shape[1])
    thr=np.percentile(matrix_data,percent*100)
    win=(win_n,win_n*ratio)
    #    if np.max(matrix_data.shape)<3000:
    #        thr=0
    #    elif np.max(matrix_data.shape)<5000:
    #        thr=np.percentile(matrix_data,95)
    #    else:
    #        thr=np.percentile(matrix_data,99)
    M_density,DEN_Dict=_generate_density_con(matrix_data,win,thr)
    #step 2.2
    NDP_Dict,ASS_Dict=_find_highpoints_v2(DEN_Dict,ratio)
    Thr_den=np.percentile(pd.Series(DEN_Dict),20)
    #step 2.3
    point_assign,class_num,centers=_assign_class(DEN_Dict,NDP_Dict,ASS_Dict,Thr_den,thr_dis)
    #step 2.4: 
    DF=_get_region_den2(DEN_Dict,NDP_Dict,ASS_Dict,point_assign,win,centers)
    #step 2.5:
    Results=_get_region_piont2(matrix_data,DF,win,centers)
    
    return Results
        
def _return_clures(DF,centers,corrbound,num_clu):
    start={}
    end={}
    #flag
    old_item=0
    for i,item in enumerate(DF['point_assign']):
        if item!=old_item:
            if old_item !=0:
                end[old_item]=i
            if item!=0 and (item not in start):
                start[item]=i
            old_item=item
    if old_item!=0:
        end[item]=i 
    #print np.max(pd.Series(end)-pd.Series(start))
    clu_count=1 
    i=0
    #item=corrbound[i]
    while (i<len(corrbound) and clu_count<len(centers)):
        item=corrbound[i] 
        if (item>centers[clu_count-1] and item<centers[clu_count]):
            if (start[clu_count+1]-item<=10) and start[clu_count+1]-end[clu_count]<2:
                end[clu_count]=item
                start[clu_count+1]=item
            clu_count=clu_count+1
            i=i+1
        elif item>centers[clu_count]:
            clu_count=clu_count+1
        else:
            i=i+1
        #print i,clu_count
    clures=pd.DataFrame({'Start':start,'End':end}, columns=['Start','End']) 
    print (np.max(clures['End']-clures['Start']))
    return clures


def _plot_HiC(matrix_data,vmax,colors=['white','red']):
    #vmax=thr
    red_list=list()
    green_list=list()
    blue_list=list()
    #colors=['white','red']
    #colors=['darkblue','green','gold','darkred']
    for color in colors:
        col=Color(color).rgb
        red_list.append(col[0])
        green_list.append(col[1])
        blue_list.append(col[2])
    c = Colormap()
    d=  {   'blue': blue_list,
            'green':green_list,
            'red':red_list}
    mycmap = c.cmap(d)
    fig,ax=plt.subplots(figsize=(8,8))
    #new_data=np.triu(matrix_data)
    #new_data=np.transpose(new_data[:,::-1])
    #mask = np.zeros_like(new_data)
    #mask[np.tril_indices_from(mask,-1)] = True
    #mask=np.transpose(mask[:,::-1])
    #with sns.axes_style("white"):
    #sns.heatmap(new_data,xticklabels=100,yticklabels=100,mask=mask,cmap=mycmap,cbar=False)
    #ax.set_facecolor('w')
    #fig.patch.set_facecolor('w')
    ax.set_facecolor('w')
    ax.grid(b=None)
    sns.heatmap(matrix_data,vmax=vmax,xticklabels=100,yticklabels=100,cmap=mycmap,cbar=False)
    #sns.heatmap(np.transpose(matrix_data[:,::-1]),vmax=vmax,xticklabels=100,yticklabels=100,cmap=mycmap,cbar=False)


def _show_diagonal_result(clures,matrix_data,thr,colors=['white','red']):
    #matrix_data[matrix_data>thr]=thr
    _plot_HiC(matrix_data,thr,colors)
    for i in range(len(clures)):
        start=clures.ix[i+1,'Start']
        end=clures.ix[i+1,'End']
        #x=[start+0.5,start+0.5,end+0.5]
        #y=[start+0.5,end+0.5,end+0.5]
        x=[start+0.5,start+0.5,end+0.5,end+0.5,start+0.5]
        y=[start+0.5,end+0.5,end+0.5,start+0.5,start+0.5]
        plt.plot(x,y,'-',color='k',lw=3)
    plt.grid(b=None)
    #plt.text(800,200,str(len(clures))+' Clusters\nThr_value (distance)= '+str(Thr_value))
    plt.show()

   

def _show_chic_clusterresult2(Results,matrix_data):
    
    if np.max(matrix_data.shape)<3000:
        thr=np.percentile(matrix_data,99.5)
    else:
        thr=np.percentile(matrix_data,99.9)
    matrix_data[matrix_data>thr]=thr
    print (thr)
    red_list=list()
    green_list=list()
    blue_list=list()
    #['darkblue','seagreen','yellow','gold','coral','hotpink','red']
    for color in ['white','green','blue','red']:
        col=Color(color).rgb
        red_list.append(col[0])
        green_list.append(col[1])
        blue_list.append(col[2])
    c = Colormap()
    d=  {   'blue': blue_list,
            'green':green_list,
            'red':red_list}
    mycmap = c.cmap(d)
    #plt.subplots(figsize=(8,8))
    #sns.heatmap(np.transpose(matrix_data[:,::-1]),cmap=mycmap)
    #sns.heatmap(np.transpose(matrix_data[:,::-1]),xticklabels=100,yticklabels=500,cmap=mycmap)
    plt.subplots(figsize=(8,8))
    #sns.heatmap(np.transpose(matrix_data[:,::-1]),cmap=mycmap)
    sns.heatmap(matrix_data.T,xticklabels=100,yticklabels=500,cmap=mycmap,cbar=False)
    #sns.heatmap(np.transpose(matrix_data[:,::-1]),xticklabels=100,yticklabels=500,cmap=mycmap,cbar=False)
    for i in Results.index:
        upper=Results.ix[i,'upper']-0.5
        bottom=Results.ix[i,'bottom']-0.5
        left=Results.ix[i,'left']-0.5
        right=Results.ix[i,'right']-0.5
        y_loc=[upper,upper,bottom,bottom,upper]
        x_loc=[left,right,right,left,left]
        plt.plot(x_loc,y_loc,'-',color='k',lw=2.5)
    plt.grid(b=None)
    plt.show()


def _show_chic_clusterresult1(Results,matrix_data):
    if np.max(matrix_data.shape)<3000:
        thr=np.percentile(matrix_data,99.5)
    else:
        thr=np.percentile(matrix_data,99.9)
    matrix_data[matrix_data>thr]=thr
    print (thr)
    red_list=list()
    green_list=list()
    blue_list=list()
    #['darkblue','seagreen','yellow','gold','coral','hotpink','red']
    for color in ['darkblue','green','yellow','gold','darkred']:
        col=Color(color).rgb
        red_list.append(col[0])
        green_list.append(col[1])
        blue_list.append(col[2])
    c = Colormap()
    d=  {   'blue': blue_list,
            'green':green_list,
            'red':red_list}
    mycmap = c.cmap(d)
    #plt.subplots(figsize=(8,8))
    #sns.heatmap(np.transpose(matrix_data[:,::-1]),cmap=mycmap)
    #sns.heatmap(np.transpose(matrix_data[:,::-1]),xticklabels=100,yticklabels=500,cmap=mycmap)
    plt.subplots(figsize=(8,8))
    #sns.heatmap(np.transpose(matrix_data[:,::-1]),cmap=mycmap)
    sns.heatmap(matrix_data,xticklabels=100,yticklabels=500,cmap=mycmap,cbar=False)
    #sns.heatmap(np.transpose(matrix_data[:,::-1]),xticklabels=100,yticklabels=500,cmap=mycmap,cbar=False)
    for i in Results.index:
        upper=Results.ix[i,'upper']-0.5
        bottom=Results.ix[i,'bottom']-0.5
        left=Results.ix[i,'left']-0.5
        right=Results.ix[i,'right']-0.5
        y_loc=[upper,upper,bottom,bottom,upper]
        x_loc=[left,right,right,left,left]
        plt.plot(x_loc,y_loc,'-',color='k',lw=2.5)
    plt.grid(b=None)
    plt.show()


def _generate_input_data(Matrix_file):
    'generate input example data for chic'
    #loc_add='H:\Dataset\Capture Hi-C\Celltypes_blood_17_location'
    #loc_add='~/.MSTDlib_test_v2/examples/Celltypes_blood_17_location'
    #Dir=os.path.dirname(MSTD.MSTDlib_test_v2.__file__)
    Dir='./src/MSTDlib'
    loc_add=Dir+'/data/Celltypes_blood_17_location'
    CHR=Matrix_file.split("_")[-1]
    #关于此染色体对应的
    pro_oe_loc=pd.read_table(loc_add+'\\'+CHR,index_col=None)
    pro_list=pro_oe_loc.ix[pro_oe_loc['type']=='promoter','loc']
    p_orderlist=sorted(pro_list)
    #p_strlist=[str(item) for item in p_orderlist]
    oe_list=pro_oe_loc.ix[pro_oe_loc['type']=='OE','loc']
    oe_orderlist=sorted(oe_list)
    oe_strlist=[str(item) for item in oe_orderlist]
    #p_orderlist=np.array(pro_list)
    fin=open(Matrix_file,'r')
    header=fin.readline()
    line=fin.readline()
    Templine=line.rstrip("\n").split("\t")
    matrix_data=np.zeros((len(p_orderlist),len(oe_orderlist)))
    #p_oe_peaks=np.zeros(len(oe_list))
    pi=0
    matrix_data[pi,oe_strlist.index(Templine[1])]=float(Templine[2])
    #当前promoter所处的位置
    promoter=Templine[0]
    for line in fin:
        Templine=line.rstrip("\n").split("\t")
        if promoter != Templine[0]:
            pi=pi+1
            #print pi
            #p_oe_peaks=np.zeros(len(oe_list))
            promoter=Templine[0]
        matrix_data[pi,oe_strlist.index(Templine[1])]=float(Templine[2])
    fin.close()
    matrix_data[np.isnan(matrix_data)]=0
    return matrix_data



        
def MSTD(Matrix_file,Output_file,MDHD=7,symmetry=1,window=10,visualization=1):
    """
    @parameters 
    Matrix_file: Input file address, the format of the file is N*N matrix file without row and column names for Hi-C maps and 
    the format of the file is N*M matrix file without row and column names for promoter capture Hi-C maps.
    Output_file: Output file address, each line in the file is triple containing boundares and centers of detected domains.
    MDHD: integer, the threshold for the minimum distance of the elements that have higher density than the element k.
    symmetry: 1/0, 1 represents the detecting of TADs and 0 represents the detecting PADs
    visualization: if visulization=1, Visualization of results can be showed.
    reso: data resolution of input data.
    """     
    if symmetry==1:
        print("#########################################################################")
        print("Step 0 : File Read ")
        print("#########################################################################")
        matrix_data=np.loadtxt(Matrix_file)
        matrix_data[np.isnan(matrix_data)]=0
        thr=np.percentile(matrix_data,99.99)
        matrix_data[matrix_data>thr]=thr      
        print("Step 0 : Done !!")
        print("#########################################################################")
        print("Step 1: define domain Only diagonal line")
        print("#########################################################################")
        DF,centers,corrbound,num_clu=_domain_only_diagonal(matrix_data,window,MDHD)
        #output cluster result
        clures=_return_clures(DF,centers,corrbound,num_clu)
        #clures=pd.DataFrame({'Start':start,'End':end,'Cen':center}, columns=['Start','End','Cen']) 
        centers=pd.DataFrame(centers, index=clures.index, columns=['Cen'])
        boundaries=pd.concat([clures,centers],axis=1)
        boundaries.to_csv(Output_file,sep='\t',index=False)
        #Output_file_center=Output_file+'_centers'
        #pd.Series(centers).to_csv(Output_file_center,index=False)
        #show results
        if visualization==1:
            #_show_diagonal_result(clures,matrix_data)
            thr=np.percentile(matrix_data,99.5)
            #thr=10
            colors=['white','green','red']
            sns.set_style("ticks")
            _show_diagonal_result(clures,matrix_data,thr,colors)
    if symmetry==2:
        print("#########################################################################")
        print("Step 0 : File Read ")
        print("#########################################################################")
        matrix_data=np.loadtxt(Matrix_file)
        thr=np.percentile(matrix_data,99.9999)
        matrix_data[matrix_data>thr]=thr
        print("Step 0 : Done !!")
        print("#########################################################################")
        print("Step 2: define structure moudle on all points or Capture Hi-C")
        print("#########################################################################")
        Results=_def_strmouOnCHiC(matrix_data,window,MDHD)
        Results.to_csv(Output_file,index=False,sep='\t')
        #Results=pd.read_table(Output_file,index_col=None)
        if visualization==1:
            #show capture hic cluster result
            _show_chic_clusterresult2(Results,matrix_data)
            #show_chic_clusterresult(centers,bound,matrix_data)

def MSTD2(Matrix_file,Output_file,MDHD=7,symmetry=1,window=5,visualization=1):
    """
    @parameters 
    Matrix_file: Input file address, the format of the file is N*N matrix file without row and column names for Hi-C maps,
    and the format of the file is triples for promoter capture Hi-C maps, 
    Output_file: Output file address, each line in the file is triple containing boundares and centers of detected domains.
    MDHD: integer, the threshold for the minimum distance of the elements that have higher density than the element k.
    symmetry: 1/0, 1 represents the detecting of TADs and 0 represents the detecting PADs
    visualization: if visulization=1, Visualization of results can be showed.
    res  
    """    
    if symmetry==1:
        print("#########################################################################")
        print("Step 0 : File Read ")
        print("#########################################################################")
        matrix_data=np.loadtxt(Matrix_file)
        matrix_data[np.isnan(matrix_data)]=0
        thr=np.percentile(matrix_data,99.99)
        matrix_data[matrix_data>thr]=thr      
        print("Step 0 : Done !!")
        print("#########################################################################")
        print("Step 1: define domain Only diagonal line")
        print("#########################################################################")
        DF,centers,corrbound,num_clu=_domain_only_diagonal(matrix_data,window,MDHD)
        #output cluster result
        clures=_return_clures(DF,centers,corrbound,num_clu)
        #clures=pd.DataFrame({'Start':start,'End':end,'Cen':center}, columns=['Start','End','Cen']) 
        centers=pd.DataFrame(centers, index=clures.index, columns=['Cen'])
        boundaries=pd.concat([clures,centers],axis=1)
        boundaries.to_csv(Output_file,sep='\t',index=False)
        #show results
        if visualization==1:
            thr=np.percentile(matrix_data,99.5)
            #thr=10
            colors=['white','green','red']
            sns.set_style("ticks")
            _show_diagonal_result(clures,matrix_data,thr,colors)
    if symmetry==2:
        print("#########################################################################")
        print("Step 0 : File Read ")
        print("#########################################################################")
        matrix_data=_generate_input_data(Matrix_file)
        #matrix_data=np.loadtxt(Matrix_file)
        thr=np.percentile(matrix_data,99.9999)
        matrix_data[matrix_data>thr]=thr
        print("Step 0 : Done !!")
        print("#########################################################################")
        print("Step 2: define structure moudle on all points or Capture Hi-C")
        print("#########################################################################")
        Results=_def_strmouOnCHiC(matrix_data,window,MDHD)
        Results.to_csv(Output_file,index=False,sep='\t')
        #Results=pd.read_table(Output_file,index_col=None)
        if visualization==1:
            #show capture hic cluster result
            _show_chic_clusterresult2(Results,matrix_data)
            #show_chic_clusterresult(centers,bound,matrix_data)

def example(symmetry=1):
    #Dir=os.getcwd()
    Dir='./src/MSTDlib'
    print("# 1. symmetry Hi-C") 
    print("# 2. asymmetry capture Hi-C")
    if symmetry==1:
        Matrix_file=Dir+'\\data\\cortex_chr6_2350-2500_HiC'
        Output_file=Dir+'\\data\\cortex_chr6_output'
        MSTD(Matrix_file,Output_file,MDHD=10,symmetry=1,window=5,visualization=1)
    elif symmetry==2:
        #example two
        Matrix_file=Dir+'\\data\\nB_chr19_480-700_CHiC'
        Output_file=Dir+'\\data\\nB_chr19_480-700_CHiC_output'
        MSTD(Matrix_file,Output_file,MDHD=100,symmetry=2,window=5,visualization=1)
    return 0








    
               











