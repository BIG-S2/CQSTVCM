setwd('./Results')


############# Figure 5 ###############
taus <- c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)

results_all<- NULL

for(tau in taus){
  est_testing <- read.csv(paste0('Est_testing_',tau,'.csv'), header=T)
  results_all <- rbind( results_all, c(est_testing$QTE_boot[2], est_testing$QDE_boot[2], est_testing$QIE_boot[2], est_testing$QDE_normal[2], est_testing$QDE_QIE_value)  )
}

colnames(results_all) <- c('QTE_boot', 'QDE_boot', 'QIE_boot', 'QDE_normal', 'QDE','QIE')

results_all <- as.data.frame(results_all)
results_all$tau=taus

write.csv(results_all, './plots/results_all.csv', row.names = F)

new_data <- data.frame('pvalues'=c( results_all$QDE_normal, results_all$QIE_boot),
                       'Effects'=c( rep('CQDE',length(taus)), rep('CQIE', length(taus))),
                       'taus'=c(taus, taus))

est_data <- data.frame('values'=c(results_all$QDE, results_all$QIE),
                       'Effects'=c( rep('CQDE',length(taus)), rep('CQIE', length(taus))),
                       'taus'=c(taus,  taus))

pic_results = ggplot(new_data, aes(x= taus, y=pvalues, group=Effects, col=Effects ))+geom_line(size=2)+ ggtitle('pvalues')+
  labs( x=expression(tau), y='pvalues across quantiles') +
  geom_hline(yintercept = 0.05, linetype='dashed')+
  scale_x_continuous(breaks = seq(0, 1, 0.1), labels = seq(0, 1, 0.1))+
  theme(
    plot.title = element_text(size = 20,hjust = 0.5), #face=bold, hjust to center the title
    legend.title = element_text(size = 15),
    legend.key = element_blank(),
    legend.key.width = unit(2, "cm"),  ### size of the legend
    legend.text = element_text(size = 20),
    legend.position = c(0.85,0.85),
    axis.title.x = element_text(size = 20),
    axis.text.x  = element_text(size = 20, angle = 0),
    axis.title.y = element_text(size = 20),
    axis.text.y  = element_text(size = 20)
  ) 


pic_QDE = ggplot(results_all, aes(x= taus, y=QDE,col='red' ))+geom_line(size=2)+ ggtitle('CQDEs across quantiles')+
  labs( x=expression(tau), y='CQDE') +
  scale_x_continuous(breaks = seq(0, 1, 0.1), labels = seq(0, 1, 0.1))+
  theme(
    plot.title = element_text(size = 20,hjust = 0.5), #face=bold, hjust to center the title
    legend.title = element_text(size = 15),
    legend.key = element_blank(),
    legend.key.width = unit(2, "cm"),  ### size of the legend
    legend.text = element_text(size = 20),
    legend.position='none',
    axis.title.x = element_text(size = 20),
    axis.text.x  = element_text(size = 20, angle = 0),
    axis.title.y = element_text(size = 20),
    axis.text.y  = element_text(size = 20)
  ) 
  

pic_QIE = ggplot(results_all, aes(x= taus, y=QIE ))+geom_line(size=2, col='#00BFC4')+ ggtitle('CQIEs across quantiles')+
  labs( x=expression(tau), y='CQIE') +
  scale_x_continuous(breaks = seq(0, 1, 0.1), labels = seq(0, 1, 0.1))+
  theme(
    plot.title = element_text(size = 20,hjust = 0.5), #face=bold, hjust to center the title
    legend.title = element_text(size = 15),
    legend.key = element_blank(),
    legend.key.width = unit(2, "cm"),  ### size of the legend
    legend.text = element_text(size = 20),
    legend.position='none',
    axis.title.x = element_text(size = 20),
    axis.text.x  = element_text(size = 20, angle = 0),
    axis.title.y = element_text(size = 20),
    axis.text.y  = element_text(size = 20)
  ) 

png.name.results <-paste0('./plots/pvalues_results.png' )
png.name.QDE <-paste0('./plots/QDE_results.png' )
png.name.QIE <-paste0('./plots/QIE_results.png' )

png(png.name.results, width=18,height=14,units="cm", res=200)
print(pic_results)
dev.off()


png(png.name.QDE, width=18,height=14,units="cm", res=200)
print(pic_QDE)
dev.off()

png(png.name.QIE, width=18,height=14,units="cm", res=200)
print(pic_QIE)
dev.off()

