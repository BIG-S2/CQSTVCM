setwd('./All_results')

###### Figure 7    ######
library(ggplot2)

effect <- 'QTE'

results_temp <- read.csv( 'All_results_final.csv', header=T )

TIs <- c(1,3)
NNs <- c(20,40)
taus <- c(0.2, 0.5, 0.8)

deltas <- c(0, 0.01, 0.025, 0.05, 0.075, 0.1)

methods <- c("Proposed", "ATE", "NoInference")


library(dplyr)

# Replace values
results_temp  <- results_temp  %>%
  mutate(method = recode(method, "Quantile" = "Proposed", "adhoc" = "NoInference", "Linear" = "ATE"))


for(NN in NNs){
  for(tau in taus){
    for(TI in TIs){
      
      select_position <- intersect( intersect(which( results_temp$NN==NN ),  which( results_temp$tau==tau ) ), which(results_temp$TI==TI) )
      
      select_data <- results_temp [select_position, ]
      
      signal <- rep(deltas, dim(select_data)[1]  )
      
      method_all <- rep(select_data$method, each = length(deltas))
      
      ddd <- data.frame( Rate = as.numeric( t(select_data [, c(5:10)] )), Signal =signal,
                         Days = days, method=method_all)
      
      pic <- ggplot( ddd, aes( x=Signal, y=Rate, color=method ) ) +
        ggtitle(paste0('Quantile Level: ', tau)) +
        labs(x="Signal", y=paste0('TI=', TI))+geom_line(size=2 )+
        geom_hline(yintercept = 0.05, linetype='dashed')+
        theme(
          plot.title = element_text(size = 20,hjust = 0.5), #face=bold, hjust to center the title
          legend.title = element_text(size = 12),
          legend.key = element_blank(),
          #legend.key.width = unit(1.8, "cm"),  ### size of the legend
          legend.text = element_text(size = 12),
          legend.position = c(0.13,0.87),
          axis.title.x = element_text(size = 20),
          axis.text.x  = element_text(size = 20, angle = 0),
          axis.title.y = element_text(size = 20),
          axis.text.y  = element_text(size = 20)
        )
      png_name <- paste0('./Simu_plots/', 'Compare_tau_', tau, '_TI_', TI,'_NN_', NN, '.png')
      
      png(png_name, width=16,height=14,units="cm", res=200)
      print(pic)
      dev.off()
      
    }
  }
}





