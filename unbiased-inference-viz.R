library(tidyverse)
library(ggthemes)
library(ggnewscale)

ugrid <- read_csv("results/sig-pairs-2-4.csv", col_names=c("t1", "t2", "p", "NSSE", "SIG"))
(uofp <- read_csv("results/uofp-2-4.csv", col_names=c("t1", "t2", "p", "NSSE", "SIG", "rep")))

true_inf = c(0.01, 0.014638605658636875, 0.02132285968115032, 0.030838618214617784, 0.044152523056469065, 0.06233071508589037, 0.08632613673629057, 0.11660186748760469, 0.15261776071192676, 0.19241575326711213, 0.2327043667374249, 0.2694432856355053, 0.2990655610946782, 0.31944614308246655, 0.3300777564019198, 0.3318781436789424, 0.32658060555840096, 0.31617137346015495, 0.30192521140643286, 0.28508701131786807, 0.26692612650643693, 0.2486333061686336, 0.2305731567719061, 0.21304215516187797, 0.19627914447801675, 0.18041615870711825, 0.16551858509849762, 0.1516194166143527, 0.13871949301080383, 0.12678811131974774, 0.11578251669241171, 0.10565571970430164, 0.09635630524682849, 0.08783110417130054, 0.08002520075710373, 0.07288402902334044, 0.06635770912211458, 0.060399007320931654, 0.05496287468140444, 0.050006447059253015, 0.04548904510430497, 0.041372186825183364, 0.03762125099464898, 0.03420579560785866, 0.031097195653568818, 0.02826870795756036, 0.02569547118263881, 0.02335450582863417, 0.021224672227985006, 0.019287260577665662, 0.01752575200110127, 0.015924706427936957, 0.014469723148510695, 0.013147440813853262, 0.011945537435688316, 0.010852760374108397, 0.009859445443688098, 0.008956827042401036, 0.008136708863410452, 0.007391551294359577, 0.006714471417371571)

infpt <- tibble(
    t = unique(ugrid$t1),
    infpt = true_inf[t] * 40
)

infln <- tibble(
    t = 1:max(ugrid$t1),
    infln = true_inf[t] * 40
)

ugrid <- bind_rows(ugrid, tibble(t1 = c(1, unique(ugrid$t2)), t2 = c(1, unique(ugrid$t2)), p=NA, NSSE=NA, SIG=NA))

altugrid <- tibble(
    t1 = ugrid$t2,
    t2 = ugrid$t1,
    NSSE = ugrid$NSSE,
    SIG = ugrid$SIG
)

lower <- function(x, a=3) pmax(x - a, 0)

upper <- function(x, t, a=5) ifelse(t < 10, x+1, x+a)

ggplot(ugrid, aes(t1, t2, fill=SIG)) +
    geom_tile() +
    scale_fill_viridis_c() +
    new_scale_fill() +
    geom_tile(aes(t1, t2, fill=NSSE), altugrid) +
    scale_fill_viridis_c(option = "magma") +
    geom_line(aes(t, infln), data=infln, col="orange", inherit.aes = FALSE, size=0.6) +
    geom_line(aes(infln, t), data=infln, col="orange", inherit.aes = FALSE, size=0.6, orientation = "y") +
    geom_linerange(
        aes(t, infpt, ymin=lower(infpt), ymax=upper(infpt, t)), 
        data=infpt, col="orange", inherit.aes = FALSE, linetype="dotted"
    ) +
    geom_linerange(
        aes(infpt, t, xmin=lower(infpt), xmax=upper(infpt, t)),
        data=infpt, col="orange", inherit.aes = FALSE, linetype="dotted"
    ) +
    scale_x_continuous(expand = expansion()) +
    scale_y_continuous(expand = expansion()) +
    labs(x="observation time 1", y="observation time 2") +
    theme_bw() +
    theme(plot.background=element_rect(color="gray60"))

ggsave("figs/sig-pairs2.pdf", width=5.4, height=4.4)

tuofp <- mutate(uofp, NSSE = NSSE * 100 + 2.2)

tuofp |>
    group_by(p) |>
    summarise(across(c(NSSE, SIG), mean)) |>
    pivot_longer(c(NSSE, SIG)) |>
    ggplot(aes(p, value, col=name)) +
    geom_point() +
    scale_x_reverse() +
    scale_y_continuous(sec.axis=sec_axis(~(. - 2.2) / 100, "NSSE")) +
    labs(x="Prop. tests at t=10", y="SIG") +
    theme_bw()

ggplot(tuofp, aes(p, NSSE)) +
    geom_point()
