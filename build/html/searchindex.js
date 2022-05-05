Search.setIndex({docnames:["index","sections/collect","sections/dataset","sections/development","sections/generate"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["index.rst","sections/collect.rst","sections/dataset.rst","sections/development.rst","sections/generate.rst"],objects:{"":[[1,0,0,"-","collect"]],"collect.generate":[[4,1,1,"","AbstractDataCollector"],[4,1,1,"","DataCollector"],[4,3,1,"","create_semantic_lidar_blueprint"],[2,0,0,"-","dataset"],[4,0,0,"-","label"],[4,0,0,"-","map"]],"collect.generate.DataCollector":[[4,2,1,"","destroy"],[4,2,1,"","parse_image"],[4,2,1,"","stop_sensor"]],"collect.generate.dataset":[[2,1,1,"","CrossValidationSplitCreator"],[2,1,1,"","SampleGroupCreator"],[2,0,0,"-","trajectron"]],"collect.generate.dataset.CrossValidationSplitCreator":[[2,2,1,"","make_splits"]],"collect.generate.dataset.SampleGroupCreator":[[2,2,1,"","make_groups"]],"collect.generate.dataset.trajectron":[[2,1,1,"","FrequencyModificationConfig"],[2,1,1,"","TrajectronSceneData"],[2,3,1,"","max_curvature"],[2,3,1,"","trajectory_curvature"]],"collect.generate.dataset.trajectron.FrequencyModificationConfig":[[2,2,1,"","from_file"]],"collect.generate.dataset.trajectron.TrajectronSceneData":[[2,4,1,"","cached_map_data"],[2,2,1,"","inspect_node"],[2,2,1,"","log_node_count"],[2,4,1,"","map_nodeids_dict"],[2,4,1,"","n_nodes"],[2,4,1,"","nodeattr_df"],[2,4,1,"","nodeid_node_dict"],[2,4,1,"","nodeid_scene_dict"],[2,4,1,"","nodeid_sls_dict"],[2,2,1,"","sample_scenes"],[2,4,1,"","scene_count_df"],[2,4,1,"","sceneattr_count_df"],[2,4,1,"","sceneid_count_dict"],[2,4,1,"","sceneid_scene_dict"],[2,4,1,"","scenes"],[2,2,1,"","set_node_fm"],[2,4,1,"","total_counts"]],"collect.generate.label":[[4,1,1,"","BoundingRegionLabel"],[4,1,1,"","SampleLabelFilter"],[4,1,1,"","SampleLabelMap"],[4,1,1,"","ScenarioIntersectionLabel"],[4,1,1,"","ScenarioSlopeLabel"],[4,1,1,"","SegmentationLabel"]],"collect.generate.label.SampleLabelFilter":[[4,2,1,"","contains"]],"collect.generate.map":[[4,1,1,"","CachedMapData"],[4,1,1,"","IntersectionReader"],[4,1,1,"","Map10HDBoundTIntersectionReader"],[4,1,1,"","MapData"],[4,1,1,"","MapDataExtractor"],[4,1,1,"","MapQuerier"],[4,1,1,"","NaiveMapQuerier"]],"collect.generate.map.CachedMapData":[[4,4,1,"","map_datum"],[4,4,1,"","map_to_scircles"],[4,4,1,"","map_to_smpolys"],[4,2,1,"","save_map_data_to_cache"]],"collect.generate.map.IntersectionReader":[[4,2,1,"","at_intersection_to_label"],[4,2,1,"","at_slope_to_label"],[4,4,1,"","wp_is_sloped"]],"collect.generate.map.Map10HDBoundTIntersectionReader":[[4,2,1,"","at_bounding_box_to_label"]],"collect.generate.map.MapData":[[4,4,1,"","road_polygons"],[4,4,1,"","white_lines"],[4,4,1,"","yellow_lines"]],"collect.generate.map.MapDataExtractor":[[4,2,1,"","extract_junction_with_portals"],[4,2,1,"","extract_road_polygons_and_lines"],[4,2,1,"","extract_waypoint_points"]],"collect.generate.map.MapQuerier":[[4,2,1,"","at_bounding_box_to_label"],[4,2,1,"","at_intersection_to_label"],[4,2,1,"","at_slope_to_label"],[4,2,1,"","curved_road_segments_enclosure_from_actor"],[4,2,1,"","render_map"],[4,2,1,"","road_boundary_constraints_from_actor"],[4,2,1,"","road_segment_enclosure_from_actor"]],collect:[[4,0,0,"-","generate"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","attribute","Python attribute"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:attribute"},terms:{"0":[3,4],"05":4,"1":[2,3,4],"11":3,"13":3,"2":4,"20":3,"2d":4,"3":3,"35":4,"4":4,"5":4,"6":3,"60":4,"7":3,"8":3,"9":3,"abstract":4,"case":3,"class":[2,4],"float":4,"function":4,"int":[2,4],"return":[2,4],"static":4,A:4,If:[2,4],Is:[],NOT:3,One:4,The:[2,4],These:2,_type:4,about:4,abstractdatacollector:4,access:4,across:2,actor:4,after:4,agent:2,all:[2,4],along:4,altern:3,an:[2,3,4],ani:2,appli:2,approx:2,ar:[2,3,4],architectur:2,around:4,arrai:4,at_bounding_box_to_label:4,at_intersection_to_label:4,at_slope_to_label:4,attrdict:[2,4],attribut:[2,4],autopilot:3,avail:2,ax:4,axi:4,b:4,base:[2,4],becom:3,befor:4,belong:2,between:4,blueprint:4,bool:4,both:3,bound:4,boundari:4,bounding_typ:4,boundingregionlabel:4,box:4,breakdown:2,builder:4,bulk:2,cach:[2,4],cached_map_data:2,cachedmapdata:[2,4],callback:4,can:4,car:[3,4],card:3,carla:[3,4],carla_map:4,carla_world:4,categor:4,center:4,certain:[3,4],chang:3,check:4,choic:4,circl:4,classifi:2,classmethod:2,client:4,close:4,code:2,codebas:3,collect:[2,3,4],collector:4,com:4,combin:[],common:3,compat:3,complete_intersect:2,comput:2,config:2,config_path:2,consist:4,constraint:4,construct:[2,4],constructor:4,contain:4,convolut:2,correspond:4,count:2,coupl:3,cover:4,cplex:3,creat:[2,4],create_semantic_lidar_blueprint:4,cross:2,crossvalidationsplitcr:2,cubic:2,cubicsplin:4,current:[2,4],curv:4,curvatur:2,curved_road_segments_enclosure_from_actor:4,data:[2,3,4],datacollector:4,datafram:2,dataset:[0,3],date:3,debug:4,deprec:4,destroi:4,develop:0,dict:[2,4],differ:4,dill:3,dim:4,disk:4,distanc:4,divid:4,doc:3,docplex:3,doe:[2,3,4],don:4,dropoff:4,e:[3,4],each:[2,4],en:3,enclosur:4,encount:4,endpoint:4,enter:4,entir:4,entranc:4,enumer:4,episod:[2,4],etc:4,evenli:2,exclud:4,exclude_sampl:4,exist:4,exit:4,extent:4,extract:4,extract_junction_with_port:4,extract_road_polygons_and_lin:4,extract_spawn_point:4,extract_waypoint_point:4,f:4,factori:4,fals:4,file:2,filenam:2,filter:4,first:4,fix:[3,4],flip:4,flip_i:4,flip_x:4,fm_modif:2,follow:4,fork:4,form:4,frame:[2,4],frequenc:[2,4],frequencymodificationconfig:2,from:[2,4],from_fil:2,gb:3,gener:[0,2],geometri:2,get:4,goal:0,gpu:3,graphic:3,group:2,h5py:3,h:4,ha:4,have:4,hill:4,histogram:2,hold:4,html:3,http:[3,4],i:[3,4],id:2,ideal:3,imag:4,impov:2,index:[0,2,4],indic:4,inform:4,initi:[2,3,4],insid:4,inspect:2,inspect_nod:2,instanc:4,instead:3,interpol:4,intersect:4,intersection_typ:4,intersectionread:4,json:2,junction:4,just:2,keep:4,kei:4,label:0,lambda:4,lane:4,latest:3,length:4,lesson:0,libcarla:4,librari:3,lidar:4,like:[2,3],line:4,linestr:2,list:[2,4],live:4,load:[3,4],locat:4,log:2,log_node_count:2,lookup:4,mai:4,make:[2,4],make_group:2,make_split:2,manag:4,map10hd:4,map10hdboundtintersectionread:4,map:[0,2],map_datum:4,map_nodeids_dict:2,map_read:4,map_to_scircl:4,map_to_smpoli:4,mapdata:4,mapdataextractor:4,mapqueri:4,matlab:3,matplotlib:4,max:4,max_curvatur:2,max_dist:4,memori:3,meter:4,mock:4,model:2,modifi:2,modul:[0,2],more:[2,3],motion:4,multipli:2,multipolygon:4,n:4,n_burn_fram:4,n_group:2,n_node:2,n_sampl:2,naivemapqueri:4,name:4,ndarrai:4,nearest:4,need:4,network:4,newer:3,node:2,node_to_df:[],nodeattr_df:2,nodeid_node_dict:2,nodeid_scene_dict:2,nodeid_sls_dict:2,nois:4,non:4,none:[2,4],note:4,np:4,number:[2,4],numer:[2,3],object:[2,3,4],obtain:4,one:2,onli:3,org:3,os2:4,other:4,other_at_intersect:2,other_at_oth:2,other_vehicle_id:4,ouster:4,out:4,page:0,pair:4,paramet:[2,4],parse_imag:4,part:4,pass:4,path:4,payload:4,per:2,persist:4,pkl:3,plan:4,player_actor:4,plot:2,po:4,point:4,polygon:4,polytop:4,posit:4,possibl:3,preprocesses:4,product:4,properti:4,provid:[2,3],proxim:4,purpos:4,python:3,pytorch:3,queri:[2,4],quick:3,reach:4,read:2,record_interv:4,refactor:4,region:4,releas:[3,4],render:4,render_map:4,replac:2,repres:[2,4],represent:4,requi:3,resampl:[],resourc:4,retriev:4,reweight:2,road:4,road_boundary_constraints_from_actor:4,road_polygon:4,road_segment_enclosure_from_actor:4,roadboundaryconstraint:4,roughli:2,rout:3,row:2,s:[3,4],sake:4,same:2,sampl:[2,4],sample_id:2,sample_scen:2,samplegroupcr:2,samplelabelfilt:4,samplelabelmap:4,sampling_precis:4,save:[2,3,4],save_directori:4,save_frequ:4,save_map_data_to_cach:4,scenariointersectionlabel:4,scenarioslopelabel:4,scene:[2,4],scene_builder_cl:4,scene_config:4,scene_count_df:2,scene_histogram:2,scene_id:2,scene_interv:4,scene_to_df:[],sceneattr_count_df:2,scenebuild:4,sceneconfig:4,sceneid_count_dict:2,sceneid_scene_dict:2,scenes_to_df:[],scipi:4,search:0,second:4,segment:4,segmentationlabel:4,select:[2,4],semant:4,semanticlidarmeasur:4,sensor:4,set:2,set_node_fm:2,shape:[2,4],share:4,should:[3,4],shuffl:2,significan:2,significant_at_intersect:2,significant_at_oth:2,simul:4,sinc:3,singl:[],size:[2,4],skip:4,slope:4,slope_degre:4,slope_pitch:4,slope_typ:4,smaller:4,so:[2,3],someth:3,space:4,spawn:4,spawn_point:4,specif:[2,4],specifi:4,spline:[2,4],split:2,stabil:2,stabl:3,stage:4,start:4,start_wp:4,still:3,stop:4,stop_sensor:4,stopped_at_intersect:2,stopped_at_oth:2,store:3,str:[2,4],straight:4,submodul:0,support:3,t:[2,4],technic:0,than:2,thei:4,thi:[3,4],through:4,timestep:4,todo:4,tol:4,topolog:4,torch:3,total:2,total_count:2,track:4,trajectori:2,trajectory_curvatur:2,trajectron:[0,3,4],trajectron_scen:4,trajectronplusplusscenebuild:4,trajectronscenedata:2,tupl:4,turn:4,turn_at_oth:2,type:[2,4],up:3,upgrad:3,us:[2,3,4],use_world_posit:[],util:[2,4],v3:4,valid:2,valu:[2,4],vehicl:4,version:3,wa:3,wai:4,waypoint:4,weak_self:4,weight:2,when:[2,4],where:4,wheter:4,whether:4,white:4,white_lin:4,without:2,work:3,world:4,wp_is_slop:4,wp_locat:4,wrangl:[],wrt:4,x:[2,4],x_max:4,x_min:4,y:4,y_max:4,y_min:4,yaw:4,yellow:4,yellow_lin:4},titles:["Welcome to CARLA Collect\u2019s documentation!","Collect submodule","Dataset submodule","Development","Generate submodule"],titleterms:{architectur:0,carla:0,collect:[0,1],content:0,dataset:2,develop:3,document:0,gener:4,goal:3,indic:0,label:4,lesson:3,map:4,s:0,submodul:[1,2,4],tabl:0,technic:3,trajectron:2,welcom:0}})