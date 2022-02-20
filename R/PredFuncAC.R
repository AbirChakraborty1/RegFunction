## Custom function for linear regression


lin_reg = function(train,test,vif=5,p=0.05,target,drop=NULL)

{
  train =dplyr::select(train,-drop)
  # Drop variables by VIF and update the formula
  formula =reformulate(".",target)
  fit=lm(formula,data=train)
  top_vif_value = as.numeric(sort(car::vif(fit),decreasing = T)[1])
  top_vif_name = names(sort(car::vif(fit),decreasing = T)[1])

  while (top_vif_value> vif) {
    top_vif_name = names(sort(car::vif(fit),decreasing = T)[1])

    fit = lm(reformulate(gsub(paste0("+ ",top_vif_name),
                              "",formula(fit)[3],fixed = T),target),data=train)
    fit = lm(reformulate(gsub(paste0(top_vif_name," +"),
                              "",formula(fit)[3],fixed = T),target),data=train)


    top_vif_value = as.numeric(sort(car::vif(fit),decreasing = T)[1])
    top_vif_name = names(sort(car::vif(fit),decreasing = T)[1])
  }
  formula(fit)

  # Drop variables by P value and update the formula

  fit=step(fit)
  top_p_value= dplyr::arrange(as.data.frame(summary(fit)[4]),desc(coefficients.Pr...t..))[1,4]
  top_p_name= rownames(dplyr::arrange(as.data.frame(summary(fit)[4]),desc(coefficients.Pr...t..))[1,])

  while (top_p_value > p) {

    fit = lm(reformulate(gsub(paste0("+ ",top_p_name),
                              "",formula(fit)[3],fixed = T),target),data=train)
    fit = lm(reformulate(gsub(paste0(top_p_name," +"),
                              "",formula(fit)[3],fixed = T),target),data=train)

    top_p_value = dplyr::arrange(as.data.frame(summary(fit)[4]),desc(coefficients.Pr...t..))[1,4]
    top_p_name = rownames(dplyr::arrange(as.data.frame(summary(fit)[4]),desc(coefficients.Pr...t..))[1,])
  }

  # Predict the target

  test.pred=predict(fit,newdata=test)
  return(test.pred)
}


## custom function for logistic regression



log_reg = function(train,test,vif=10,p=0.05,target,drop=NULL)


{
  train =dplyr::select(train,-drop)

  # Drop variables by VIF and update the formula

  formula =reformulate(".",target)
  fit=lm(formula,data=train)
  top_vif_value = as.numeric(sort(car::vif(fit),decreasing = T)[1])
  top_vif_name = names(sort(car::vif(fit),decreasing = T)[1])

  while (top_vif_value> vif) {
    top_vif_name = names(sort(car::vif(fit),decreasing = T)[1])

    fit = lm(reformulate(gsub(paste0("+ ",top_vif_name),
                              "",formula(fit)[3],fixed = T),target),data=train)
    fit = lm(reformulate(gsub(paste0(top_vif_name," +"),
                              "",formula(fit)[3],fixed = T),target),data=train)


    top_vif_value = as.numeric(sort(car::vif(fit),decreasing = T)[1])
    top_vif_name = names(sort(car::vif(fit),decreasing = T)[1])
  }
  formula(fit)


  log_fit=glm(formula(fit),data=train,family = "binomial")
  log_fit=step(log_fit)
  formula(log_fit)
  summary(log_fit)

  L = as.list(summary(log_fit))
  L = as.data.frame(L[12])[4]
  L$var  = rownames(L)
  L = L[-1,]

  top_p_value= dplyr::arrange(L,desc(coefficients.Pr...z..))[1,2]
  top_p_name= dplyr::arrange(L,desc(coefficients.Pr...z..))[1,1]

  while (top_p_value > p) {

    log_fit = lm(reformulate(gsub(paste0("+ ",top_p_name),
                                  "",formula(log_fit)[3],fixed = T),target),data=train)
    log_fit = lm(reformulate(gsub(paste0(top_p_name," +"),
                                  "",formula(log_fit)[3],fixed = T),target),data=train)

    top_p_value = dplyr::arrange(as.data.frame(summary(log_fit)[4]),desc(coefficients.Pr...t..))[1,4]
    top_p_name = rownames(dplyr::arrange(as.data.frame(summary(log_fit)[4]),desc(coefficients.Pr...t..))[1,])

    test.score=car::Predict(log_fit,newdata = test,type='response')



  }
}


## custom function for decision tree

tree_reg = function(train,test,target,drop=NULL)
{
  train =dplyr::select(train,-drop)
  formula =reformulate(".",target)
  ld.tree=tree::tree(formula,data=train)

  plot(ld.tree)
  text(ld.tree)

  test.pred=predict(ld.tree,newdata = test)
  return(test.pred)

}


## Custom Function for GBM

gbm_reg = function(train,test,target,drop = NULL,interaction.depth=c(1:7),
                   n.trees=c(50,100,200,500,700),shrinkage=c(.1,.01,.001),
                   n.minobsinnode=c(1,2,5,10),num_trials=10)

{
  train =dplyr::select(train,-drop)
  param=list(interaction.depth,n.trees,shrinkage,n.minobsinnode)
  subset_paras=function(full_list_para,n=10){

    all_comb=expand.grid(full_list_para)

    s=sample(1:nrow(all_comb),n)

    subset_para=all_comb[s,]

    return(subset_para)
  }

  num_trials=10
  my_params=subset_paras(param,num_trials)

  myerror=999999999

  for(i in 1:num_trials){
    print(paste0('starting iteration:',i))
    # uncomment the line above to keep track of progress
    names(my_params)=

      params=my_params[i,]
    names(params)=c('interaction.depth','n.trees','shrinkage','n.minobsinnode')

    k=cvTools::cvTuning(gbm::gbm,reformulate(".",target),
                        data = train,
                        tuning =params,
                        args = list(distribution="gaussian"),
                        folds = cvTools::cvFolds(nrow(train), K=10, type = "random"),
                        seed =2,
                        predictArgs = list(n.trees=params$n.trees)
    )
    score.this=k$cv[,2]

    if(score.this<myerror){
      print(params)
      # uncomment the line above to keep track of progress
      myerror=score.this
      print(myerror)
      # uncomment the line above to keep track of progress
      best_params=params
    }

    print('DONE')
    # uncomment the line above to keep track of progress
  }


  ld.gbm.final=gbm::gbm(reformulate(".",target),data=train,
                        n.trees = best_params$n.trees,
                        n.minobsinnode = best_params$n.minobsinnode,
                        shrinkage = best_params$shrinkage,
                        interaction.depth = best_params$interaction.depth,
                        distribution = "gaussian")
  test.pred=predict(ld.gbm.final,newdata=test,n.trees = best_params$n.trees)
  return(test.pred)
}

## Random Forest with Parameter Tuning

rf_pt_reg = function(train,test,target,drop = NULL,mtry=c(5,10,15,20,25),
                     ntree=c(50,100,200,500,700),maxnodes=c(5,10,15,20,30,50),
                     nodesize=c(1,2,5,10),num_trials=10)
{
  train =dplyr::select(train,-drop)
  param=list(mtry,ntree,maxnodes,nodesize)

  subset_paras=function(full_list_para,n=10){

    all_comb=expand.grid(full_list_para)

    s=sample(1:nrow(all_comb),n)

    subset_para=all_comb[s,]

    return(subset_para)
  }
  my_params=subset_paras(param,num_trials)
  myerror=999999999


  for(i in 1:num_trials){
    print(paste0('starting iteration:',i))
    # uncomment the line above to keep track of progress
    params=my_params[i,]
    names(params)=c('mtry','ntree','maxnodes','nodesize')

    k=cvTools::cvTuning(randomForest::randomForest,reformulate(".",target),
                        data =train,
                        tuning =params,
                        folds = cvTools::cvFolds(nrow(train), K=10, type = "random"),
                        seed =2)

    score.this=k$cv[,2]

    if(score.this<myerror){
      print(params)
      # uncomment the line above to keep track of progress
      myerror=score.this
      print(myerror)
      # uncomment the line above to keep track of progress
      best_params=params
    }

    print('DONE')
    # uncomment the line above to keep track of progress
  }


  ld.rf.final=randomForest::randomForest(reformulate(".",target),
                                         mtry=best_params$mtry,
                                         ntree=best_params$ntree,
                                         maxnodes=best_params$maxnodes,
                                         nodesize=best_params$nodesize,
                                         data=ld_train)

  test.pred=predict(ld.rf.final,newdata=test)
  return(test.pred)
}













