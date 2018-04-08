# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:16:04 2017
@author: Yusen Ye (ccyeyusen@sina.com)
@brief: MSTD is a generic and efficient method to  identify multi-scale topological domains (MSTD) 
        from symmetric Hi-C and other high resolution asymmetric promoter capture Hi-C datasets
@version 0.0.2

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from colormap import Color, Colormap
#Data=matrix_data; distance=MDHD; win_n=5
def _domain_only_diagonal(Data,win_n,distance):
    
    Dsize=Data.shape[0]
    #step1.1
    pdensity=np.zeros(Dsize)
    DEN_Dict={}
    for ip in range(Dsize):
        begin_i=ip-win_n+1
        end_i=ip+win_n-1
        if (begin_i<=0) | (end_i>=Dsize-1):
            if begin_i<0:
                begin_i=0    
            
            if end_i>Dsize-1:
                end_i=Dsize-1     
            pdensity[ip]=np.mean(Data[begin_i:ip+1,:][:,ip:end_i+1])
            DEN_Dict[ip]=pdensity[ip]
        else:
            pdensity[ip]= pdensity[ip-1] + (-np.sum(Data[begin_i-1:ip,ip-1])-np.sum(Data[begin_i-1,ip:end_i])
            +np.sum(Data[ip,ip:end_i+1])+ np.sum(Data[begin_i:ip,end_i]))/(win_n*win_n)
            DEN_Dict[ip]=pdensity[ip]+np.random.random(1)/1000
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
    
    #boundaries DF
    start={}
    end={}
    center={}
    Thr_den=np.percentile(pdensity,20)
    point_assign={}
    for temp in DEN_Dict:
        point_assign[temp]=0
    #class_num=1
    join_num=0
    #centers=[]
    for item in DEN_Dict:
        den=DEN_Dict[item]
        dist=NDP_Dict[item]
        if ((den>Thr_den) & (dist>distance)):
            join_num=join_num+1
            point_assign[item]=join_num
            #class_num=class_num+1
            start[join_num]=item
            end[join_num]=item
            center[join_num]=item
            #centers.append(item)
            ASS_Dict[item]=item
    clures=pd.DataFrame({'Start':start,'End':end,'Cen':center}, columns=['Start','End','Cen']) 
    
    old_join_num=0
    new_join_num=join_num
    while old_join_num!=new_join_num:
        old_join_num=join_num
        for item in DEN_Dict:
            if ((NDP_Dict[item]<=distance)):
                if ASS_Dict[item]==item:
                    continue
                fclass=point_assign[ASS_Dict[item]]
                if fclass !=0:
                    mclass= point_assign[item]
                    if mclass == 0:
                        temp=center[fclass]
                        if (DEN_Dict[item]>DEN_Dict[temp]/5):
                            #判断此点是否在类别范围
                            item_class= clures[(item>clures['Start']) & (clures['End']>item)].values
                            if len(item_class)!=0:
                                point_assign[item]=point_assign[ASS_Dict[item_class[0][2]]]
                            else:
                                #print item
                                point_assign[item]=point_assign[ASS_Dict[item]]
                                if item < clures.ix[point_assign[item],'Start']:
                                    clures.ix[ point_assign[item],'Start']=item 
                                else:
                                    clures.ix[ point_assign[item], 'End']=item 
                            join_num=join_num+1
        new_join_num=join_num 
    
    step=3
    for clu in clures.index[:-1:1]:
        left=clures.loc[clu,'End']
        right=clures.loc[clu+1,'Start']
        if (left-step>=0) & (right+step<=Dsize-1):
            if left==right-1:
                loca=np.argmin(pdensity[left-step:right+step])
                newbound=left-step+loca
                clures.loc[clu,'End']=newbound
                clures.loc[clu+1,'Start']=newbound
    return clures

#Data=matrix_data
def _generate_density_con(Data,win,thr,MDHD):
    Dsize=Data.shape
    M_density=np.zeros(Dsize)
    DEN_Dict={}
    if Dsize[0]==Dsize[1]:
        for i in range(Dsize[0]):
            for j in range(Dsize[1]):
                if i-j>MDHD*4:
                    begin_i=i-win[0]
                    begin_j=j-win[1]
                    end_i=i+win[0]
                    end_j=j+win[1]
                    if (begin_i<0)| (begin_j<0)| (end_i>Dsize[0]-1)|(end_j>Dsize[1]-1):
                        if begin_i<0:
                            begin_i=0
                        if begin_j<0:
                            begin_j=0
                        if end_i>Dsize[0]-1:
                            end_i=Dsize[0]-1
                        if end_j>Dsize[1]-1:
                            end_j=Dsize[1]-1
                        M_density[i,j]=np.mean(Data[begin_i:end_i,begin_j:end_j])+np.random.random(1)/1000.0                
                    else:  
                        M_density[i,j]=M_density[i,j-1]+ (-np.sum(Data[begin_i:end_i,begin_j-1])
                        +np.sum(Data[begin_i:end_i,end_j-1]))/(4*win[0]*win[1])
                    if Data[i,j]>thr:
                        DEN_Dict[(i,j)]=M_density[i,j] 
    else:
        for i in range(Dsize[0]):
            for j in range(Dsize[1]):
                begin_i=i-win[0]
                begin_j=j-win[1]
                end_i=i+win[0]
                end_j=j+win[1]
                if (begin_i<0)| (begin_j<0)| (end_i>Dsize[0]-1)|(end_j>Dsize[1]-1):
                    if begin_i<0:
                        begin_i=0
                    if begin_j<0:
                        begin_j=0
                    if end_i>Dsize[0]-1:
                        end_i=Dsize[0]-1
                    if end_j>Dsize[1]-1:
                        end_j=Dsize[1]-1
                    M_density[i,j]=np.mean(Data[begin_i:end_i,begin_j:end_j])+np.random.random(1)/1000.0                
                else:  
                    M_density[i,j]=M_density[i,j-1]+ (-np.sum(Data[begin_i:end_i,begin_j-1])
                    +np.sum(Data[begin_i:end_i,end_j-1]))/(4*win[0]*win[1])
                if Data[i,j]>thr:
                    DEN_Dict[(i,j)]=M_density[i,j] 
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
            step_point=[(item[0]+st,item[1]+ra) for st in np.arange(-step,step+1) for ra in np.arange(-step*ratio,step*ratio+1) 
            if (abs(st)==step or ratio*(step-1)<abs(ra)<=ratio*step)]
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
  
def _assign_class(DEN_Dict,NDP_Dict,ASS_Dict,Thr_den,Thr_dis,reso=3):
    locs=['upper','bottom','left','right','cen_x','cen_y']
    
    point_assign={}
    for temp in DEN_Dict:
        point_assign[temp]=0    
    #class_num=1
    join_num=0
    boundaries=pd.DataFrame()
    center=dict()
    
    for item in DEN_Dict:
        den=DEN_Dict[item]
        dist=NDP_Dict[item]
        #value=den*dist      
        bound=list()
        if (den>=Thr_den) and (dist>=Thr_dis):
            join_num=join_num+1
            point_assign[item]=join_num
            center[join_num]=item
            #class_num=class_num+1
            bound.append(item[0])
            bound.append(item[0]+1)
            bound.append(item[1])
            bound.append(item[1]+1)
            bound.append(item[0])
            bound.append(item[1])
            #for k in range(len(locs)):
            #   if (k<2) | (k==4):
            #       bound.append(item[0])
            #   else:
            #       bound.append(item[1])
            ASS_Dict[item]=item
            bound=pd.DataFrame(bound)
            boundaries=pd.concat([boundaries,bound.T],axis=0)
    boundaries.columns=locs
    boundaries.index=np.arange(1,len(boundaries)+1)
    
    
    Thr_den1=np.percentile(pd.Series(DEN_Dict),5)
    #Al=len(DEN_Dict)
    old_join_num=0
    new_join_num=join_num
    while old_join_num!= new_join_num:
        old_join_num=join_num
        for item in DEN_Dict:
            if NDP_Dict[item]<Thr_dis:
                if ASS_Dict[item]==item:
                    continue
                fclass=point_assign[ASS_Dict[item]]
                if fclass!=0:
                    #print item
                    mclass=point_assign[item]
                    if mclass==0:
                        temp=center[fclass]
                        #if DEN_Dict[item]>DEN_Dict[temp]/3 and (DEN_Dict[item]>Thr_den1):
                        #if DEN_Dict[item]>np.max(DEN_Dict[temp]/5,Thr_den1):
                        if (DEN_Dict[item]>Thr_den1):
                            #判断此点是否在类别范围
                            item_class=boundaries[((item[0]>boundaries['upper']) & (boundaries['bottom']>item[0]) & 
                            (item[1]>boundaries['left']) & (boundaries['right']>item[1]))].values
                            if len(item_class)>0:
                                if len(item_class)>1:
                                    print (item_class)
                                point_assign[item] = point_assign[ASS_Dict[(item_class[0][4],item_class[0][5])]]
                            else:
                                #print item, 2
                                X1=boundaries.ix[fclass,'upper']
                                X2=boundaries.ix[fclass,'bottom']
                                X3=boundaries.ix[fclass,'left']
                                X4=boundaries.ix[fclass,'right']
                                #print X1,X2,X3,X4
                                #更新前确认不能有任何重合
                                point_assign[item]=fclass
                                if item[0]<X1:
                                    sub_bound=boundaries[boundaries['bottom'] <=X1]
                                    if np.all(((sub_bound['right']<= X3) | (sub_bound['left']>= X4) | 
                                    (item[0] >= sub_bound['bottom']))):
                                        #print (item)
                                        boundaries.ix[fclass,'upper']=item[0]
                                elif item[0] > X2:
                                    sub_bound=boundaries[boundaries['upper'] >= X2]
                                    if np.all(( (sub_bound['left']>= X4) | (sub_bound['right']<=X3)) | 
                                    (item[0] <= sub_bound['upper'])):
                                        #print (item)
                                        boundaries.ix[fclass, 'bottom']=item[0]
                                if item[1]<X3:
                                    sub_bound=boundaries[boundaries['right'] <=X3]
                                    if np.all( (sub_bound['bottom']<=X1) | (sub_bound['upper']>=X2) | 
                                    (item[1] >= sub_bound['right'])):
                                        #print (item)
                                        boundaries.ix[fclass,'left']=item[1]
                                elif item[1]>X4:
                                    sub_bound=boundaries[boundaries['left'] >= X4]
                                    if np.all( (sub_bound['bottom']<=X1) | (sub_bound['upper']>=X2) | 
                                    (item[1] <= sub_bound['left'])):
                                        #print (item)
                                        boundaries.ix[fclass,'right']=item[1]
                            join_num=join_num+1
            new_join_num=join_num
    return boundaries, point_assign,center

def _def_strmouOnCHiC(matrix_data,win_n=5,thr_dis=15):
    
    #matrix size
    Mat_size=matrix_data.shape
    print ("Matrix size:"+str(Mat_size[0])+'*'+str(Mat_size[1]))   
    ratio=int(matrix_data.shape[1]//matrix_data.shape[0])
    #computing density threshold
    if ratio==1:
        #point_num=matrix_data.shape[0] * (2000000/reso)*2
        point_num=matrix_data.shape[0] * thr_dis *8
    else:
        point_num=matrix_data.shape[0]*200/6
    
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
    M_density,DEN_Dict=_generate_density_con(matrix_data,win,thr,thr_dis)
    #step 2.2
    
    NDP_Dict,ASS_Dict=_find_highpoints_v2(DEN_Dict,ratio)
    
    
    if ratio==1:
        Thr_den=np.percentile(pd.Series(DEN_Dict),90)
    else:
        Thr_den=np.percentile(pd.Series(DEN_Dict),20)
        
    #step 2.3
    boundaries,point_assign,centers=_assign_class(DEN_Dict,NDP_Dict,ASS_Dict,Thr_den,thr_dis)
    
    #Dict={'density':DEN_Dict,'distance':NDP_Dict,'ass_point':ASS_Dict,'point_assign':point_assign}
    #DF=pd.DataFrame(Dict) 
    #density_distance(DF,centers)
    #
    #thr=np.percentile(matrix_data,99)
    #
    #colors=['white','red','blue']
    #_show_chic_clusterresult2(boundaries,matrix_data,thr,colors)
    #step 2.4: 
    #DF=_get_region_den2(DEN_Dict,NDP_Dict,ASS_Dict,point_assign,win,centers)
    #step 2.5:
    #Results=_get_region_piont2(matrix_data,DF,win,centers)
    return boundaries      

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
    #print (np.max(clures['End']-clures['Start']))
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
    sns.heatmap(matrix_data.T,vmax=vmax,xticklabels=100,yticklabels=100,cmap=mycmap,cbar=False)
    #sns.heatmap(np.transpose(matrix_data[:,::-1]),vmax=vmax,xticklabels=100,yticklabels=100,cmap=mycmap,cbar=False)

def density_distance(DF,centers):
    X_center=[DF.loc[centers[line],'density'] for line in centers]
    Y_center=[DF.loc[centers[line],'distance'] for line in centers]
    fig,ax=plt.subplots(figsize=(6,6))
    T_DF=DF.loc[DF['point_assign']==0]
    Index=T_DF.index
    X=[T_DF.loc[line,'density'] for line in Index]
    Y=[T_DF.loc[line,'distance'] for line in Index]
    plt.plot(X,Y,'.',color='k',markersize=10)
    colors=sns.color_palette("Set1", n_colors=15)
    for i in range(len(centers)):
        T_DF=DF.loc[DF['point_assign']==i+1]
        Index=T_DF.index
        #print len(Index)
        X=[T_DF.loc[line,'density'] for line in Index]
        Y=[T_DF.loc[line,'distance'] for line in Index]
        plt.plot(X,Y,'.',color=colors[i%len(colors)],markersize=10)
        plt.plot(X_center[i],Y_center[i],'o',color=colors[i%len(colors)],markersize=16)
    #plt.xlabel('density',fontsize=14)
    #plt.ylabel('distance',fontsize=14)
    #ax.axes.set_xticks([])
    #ax.axes.set_yticks([])
    ax.set_facecolor('w')
    ax.grid(b=None)
    #ax.spines['bottom'].set_position(('data',0))
    #ax.spines['left'].set_position(('left'))
    plt.show()    
    
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


def _show_chic_clusterresult2(Results,matrix_data,thr,colors):
    #Results=boundaries
    #if np.max(matrix_data.shape)<3000:
    #    thr=np.percentile(matrix_data,99.5)
    #else:
    #    thr=np.percentile(matrix_data,99.9)
    #matrix_data[matrix_data>thr]=thr
    #print thr
    #matrix_data[matrix_data>thr]=thr
    #mask some points
    _plot_HiC(matrix_data,thr,colors)
    #    red_list=list()
    #    green_list=list()
    #    blue_list=list()
    #    #['darkblue','seagreen','yellow','gold','coral','hotpink','red']
    #    for color in ['white','green','blue','red']:
    #        col=Color(color).rgb
    #        red_list.append(col[0])
    #        green_list.append(col[1])
    #        blue_list.append(col[2])
    #    c = Colormap()
    #    d=  {   'blue': blue_list,
    #            'green':green_list,
    #            'red':red_list}
    #    mycmap = c.cmap(d)
    #plt.subplots(figsize=(8,8))
    #sns.heatmap(np.transpose(matrix_data[:,::-1]),cmap=mycmap)
    #sns.heatmap(np.transpose(matrix_data[:,::-1]),xticklabels=100,yticklabels=500,cmap=mycmap)
    #plt.subplots(figsize=(8,8))
    #sns.heatmap(np.transpose(matrix_data[:,::-1]),cmap=mycmap)
    #sns.heatmap(np.transpose(matrix_data[:,::-1]),xticklabels=100,yticklabels=500,cbar=False)
    for i in Results.index:
        upper=Results.ix[i,'upper']
        bottom=Results.ix[i,'bottom']
        left=Results.ix[i,'left']
        right=Results.ix[i,'right']
        y_loc=[upper,upper,bottom,bottom,upper]
        x_loc=[left,right,right,left,left]
        plt.plot(x_loc,y_loc,'-',color='k',lw=2.5)
    plt.grid(b=None)
    plt.show()


def _show_chic_clusterresult3(Results,matrix_data,colors=['white','green','blue','red']):
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
    #plt.subplots(figsize=(8,8))
    #sns.heatmap(np.transpose(matrix_data[:,::-1]),cmap=mycmap)
    #sns.heatmap(np.transpose(matrix_data[:,::-1]),xticklabels=100,yticklabels=500,cmap=mycmap)
    plt.subplots(figsize=(8,8))
    
    #sns.heatmap(np.transpose(matrix_data[:,::-1]),cmap=mycmap)
    
    #sns.heatmap(np.transpose(matrix_data[:,::-1]),xticklabels=100,yticklabels=500,cmap=mycmap,cbar=False)
    sns.heatmap(matrix_data.T,xticklabels=100,yticklabels=1000,cmap=mycmap,cbar=False)
    
    for i in Results.index:
        upper=Results.ix[i,'upper']
        bottom=Results.ix[i,'bottom']
        left=Results.ix[i,'left']
        right=Results.ix[i,'right']
        x_loc=[upper,upper,bottom,bottom,upper]
        y_loc=[left,right,right,left,left]
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
    sns.heatmap(np.transpose(matrix_data[:,::-1]),xticklabels=100,yticklabels=500,cmap=mycmap,cbar=False)
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


def mask_near(matrix_data,MDHD):
    #mask 2M values
    Dsize=matrix_data.shape
    tril_M=np.tril(matrix_data)
    
    for i in range(Dsize[0]):
        for j in range(Dsize[1]):
            if ((0<=i-j) & (i-j<=MDHD*4)):
                tril_M[i,j]=0
    return tril_M
                

                
def MSTD(Matrix_file,Output_file,MDHD=5,symmetry=1,window=5,visualization=0):
    """
    @parameters 
    Matrix_file: Input file address, the format of the file is N*N matrix file without row and column names.
    Output_file: Output file address, each line in the file is triple containing boundares and centers of detected domains.
    MDHD: integer, the threshold for the minimum distance of the elements that have higher density than the element k.
    symmetry: 1/0, 1 represents the detecting of TADs and 0 represents the detecting PADs
    visualization: if visulization=1, Visualization of results can be showed.
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
        clures=_domain_only_diagonal(matrix_data,window,MDHD)
        #output cluster result
        #clures=_return_clures(DF,centers,corrbound,num_clu)
        clures.to_csv(Output_file,sep='\t',index=False)
        #Output_file_center=Output_file+'_centers'
        #pd.Series(centers).to_csv(Output_file_center,index=False)
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
        matrix_data=np.loadtxt(Matrix_file)
        thr=np.percentile(matrix_data,99.99)
        matrix_data[matrix_data>thr]=thr
        print("Step 0 : Done !!")
        print("#########################################################################")
        print("Step 2: define structure moudle on all points or Capture Hi-C")
        print("#########################################################################")
        #MDHD=150;window=5
        boundaries=_def_strmouOnCHiC(matrix_data,window,MDHD)
        boundaries.to_csv(Output_file,index=False,sep='\t')
        #Results=pd.read_table(Output_file,index_col=None)
        
        if visualization==1:
            ratio=matrix_data.shape[1]//matrix_data.shape[0]
            if ratio==1:
                tril_M=mask_near(matrix_data,MDHD)
                thr=np.percentile(tril_M,99.5)
                colors=['white','green','red']
                #colors=['white','red']
                sns.set_style("white")
                _show_chic_clusterresult2(boundaries,tril_M.T,thr,colors)
            else:
                #show capture hic cluster result
                colors=['white','green','red']
                #colors=['white','red']
                sns.set_style("white")
                _show_chic_clusterresult3(boundaries,matrix_data,colors)
                #show_chic_clusterresult(centers,bound,matrix_data)
                
def MSTD2(Matrix_file,Output_file,MDHD=5,symmetry=1,window=5,visualization=1):
    """
    @parameters 
    Matrix_file: Input file address, the format of the file is triple for promoter capture Hi-C maps, the format of the file is
    N*N matrix file without row and column names for Hi-C maps.
    Output_file: Output file address, each line in the file is triple containing boundares and centers of detected domains.
    MDHD: integer, the threshold for the minimum distance of the elements that have higher density than the element k.
    symmetry: 1/0, 1 represents the detecting of TADs and 0 represents the detecting PADs
    visualization: if visulization=1, Visualization of results can be showed. 
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
        clures=_domain_only_diagonal(matrix_data,window,MDHD)
        #output cluster result
        #clures=_return_clures(DF,centers,corrbound,num_clu)
        clures.to_csv(Output_file,sep='\t',index=False)
        #Output_file_center=Output_file+'_centers'
        #pd.Series(centers).to_csv(Output_file_center,index=False)
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
        boundaries=_def_strmouOnCHiC(matrix_data,window,MDHD)
        boundaries.to_csv(Output_file,index=False,sep='\t')
        #Results=pd.read_table(Output_file,index_col=None)
        if visualization==1:
            ratio=matrix_data.shape[1]//matrix_data.shape[0]
            if ratio==1:
                tril_M=mask_near(matrix_data,MDHD)
                thr=np.percentile(tril_M,99.5)
                colors=['white','green','red']
                #colors=['white','red']
                sns.set_style("white")
                _show_chic_clusterresult2(boundaries,tril_M.T,thr,colors)
            else:
                #show capture hic cluster result
                colors=['white','green','red']
                #colors=['white','red']
                sns.set_style("white")
                _show_chic_clusterresult3(boundaries,matrix_data,colors)
                #show_chic_clusterresult(centers,bound,matrix_data)

def example(symmetry=1):
    #Dir=os.getcwd()
    #Dir=example_add
    Dir='./src/MSTDlib'
    #Dir=MSTDlib_test_v2.
    
    print("# 1. symmetry Hi-C") 
    print("# 2. asymmetry capture Hi-C")
    if symmetry==1:
        Matrix_file=Dir+'\\data\\cortex_chr6_2350-2500_HiC'
        Output_file=Dir+'\\data\\cortex_chr6_output'
        MSTD(Matrix_file,Output_file,MDHD=10,symmetry=1,window=10,visualization=1)
    elif symmetry==2:
        #example two
        Matrix_file=Dir+'\\data\\nB_chr19_480-700_CHiC'
        Output_file=Dir+'\\data\\nB_chr19_480-700_CHiC_output'
        MSTD(Matrix_file,Output_file,MDHD=100,symmetry=2,window=5,visualization=1)
    return 0








    
               












