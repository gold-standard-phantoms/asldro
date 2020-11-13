Ground Truth
============



hrgt_icbm_2009a_nls_v4
~~~~~~~~~~~~~~~~~~~~~~

A 1x1x1mm ground truth based on the MNI ICBM 2009a Nonlinear
Symmetric template, obtained from http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009,
with quantitative parameters set based on supplied masks.  This ground truth has the following
values for each tissue and quantity (corresponding to 3T):

+--------------+----------------+--------------+----------+----------+----------+----------+
| Tissue       | Perfusion Rate | Transit Time | T1       | T2       | T2*      | label    |
|              | [ml/100g/min]  | [s]          | [s]      | [s]      | [s]      |          |
+==============+================+==============+==========+==========+==========+==========+
| Grey Matter  | 60.00          | 1.00         | 1.33     | 0.110    | 0.050    | 1        | 
+--------------+----------------+--------------+----------+----------+----------+----------+
| White Matter | 20.00          | 1.50         | 0.83     | 0.080    | 0.050    | 2        |
+--------------+----------------+--------------+----------+----------+----------+----------+
| CSF          | 0.00           | 1000.0       | 3.00     | 0.300    | 0.200    | 3        |
+--------------+----------------+--------------+----------+----------+----------+----------+

label is the integer value assigned to the tissue in the seg_label volume.