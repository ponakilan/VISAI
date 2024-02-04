/-----------------------------------------------------\\
| ECO data set (Electricity Consumption & Occupancy)  ||
| A Research Project of the Distributed Systems Group ||
| http://www.vs.inf.ethz.ch/                          ||
| Wilhelm Kleiminger, Christian Beckel                ||
\=====================================================//

1. SUMMARY

The ECO (Electricity Consumption and Occupancy) data set is a comprehensive open-source (Creative Commons License CC BY 4.0) data set for non-intrusive load monitoring and occupancy detection research. It was collected in 6 Swiss households over a period of 8 months. For each of the households, the ECO data set provides:

* 1 Hz aggregate consumption data. Each measurement contains data on current, voltage, and phase shift for each of the three phases in the household.
* 1 Hz plug-level data measured from selected appliances.
* Occupancy information measured through a tablet computer (manual labeling) and a passive infrared sensor (in some of the households).

2. CONTEXT

The data was collected in the context of the project Smart Meter Services. The project is related to the open source framework NILM-Eval[1], in which we utilize the ECO data set to evaluate a set of non-intrusive load monitoring algorithms. We have also published the OpenWrt package Pylon[2], which allows for obtaining measurements from SML-based smart electricity meters.

3. FILES

The files are coded in the following way. The first two digits (e.g., 01-06 identify the household). The next part {sm, plugs, occupancy} identifies the type of data (i.e., readings from the smart meter, the plugs and the occupancy ground truth). Lastly, where applicable, the suffix indicates the format of the data (either Matlab or plain CSV).

4. PUBLICATIONS

The following publications contain more detailed information on the data set and the measurement infrastructure deployed into the six household to collect the data. Please *CITE ONE* or both of these papers when using the data set.

* Wilhelm Kleiminger, Christian Beckel, Silvia Santini
  Household Occupancy Monitoring Using Electricity Meters.
  Proceedings of the 2015 ACM International Joint Conference on Pervasive and Ubiquitous Computing (UbiComp 2015). Osaka,     Japan, September 2015.

* Christian Beckel, Wilhelm Kleiminger, Romano Cicchetti, Thorsten Staake, and Silvia Santini
  The ECO Data Set and the Performance of Non-Intrusive Load Monitoring Algorithms.
  Proceedings of the 1st ACM International Conference on Embedded Systems for Energy-Efficient Buildings (BuildSys 2014).     Memphis, TN, USA. ACM, November 2014.

5. ACKNOWLEDGMENTS

We thank our project partner Energie Thun and the participating households for their support. We also want to thank our students (Zeno Koller, Andreas Dröscher, Christian Stücklberger, Daniel Pauli, Dominique im Obersteg, Manuel Kläy, Michael Spiegel, Romano Cicchetti, Sarah Kilcher, Steven van Damme, and Thomas Selber) for their valuable support in collecting, analyzing, and visualizing the data.

[1] https://github.com/beckel/nilm-eval
[2] https://github.com/wkleiminger/pylon