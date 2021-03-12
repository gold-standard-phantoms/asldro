Input Ground Truth
===================

The input ground truth, known as the High Resolution Ground Truth (or HRGT)
comprises of a 5-dimensional NIFTI image, and json parameter file that
describes the NIFTI and also supplies any non-image ground truth parameters. 3D Volumes for each
ground truth quantity are concatenated along the 5th NIFTI dimension.

.. image:: /images/ground_truth_concatenation.png

Selecting a ground truth
-------------------------

Different HRGT's are selected by modifiying the "ground_truth" entry in the input parameter file to
the name of the ground truth being used.  For example:

.. code-block:: json

    {
        "global_configuration": {
            "ground_truth": "hrgt_icbm_2009a_nls_3t"
        }
    }

will use the built-in ground truth "hrgt_icbm_2009a_nls_3t" (see below for more details of these
datasets). In addition, it is possible to specify your own ground truth files by using one of the
following:

.. code-block:: json

    {
        "global_configuration": {
            "ground_truth": "/path/to/nifti_file.nii"
        }
    }

or:

.. code-block:: json

    {
        "global_configuration": {
            "ground_truth": {
                "nii": "/path/to/nifti_file.nii.gz",
                "json": "/path/to/json_file.json"
        }
    }

In the two examples above, the first example assumes there is a JSON file at precisely the same path
with the same filename, except for a '.json' extension instead of a '.nii'/'.nii.gz' extension.
The second example uses an explicitly defined filename for each file type, and may have different
paths.

Augmenting values
~~~~~~~~~~~~~~~~~

HRGT parameters and image values can be augmented using the input parameter file. See :doc:`parameters` for more information.


Custom ground truth
---------------------

ASLDRO supports the use of custom ground truths. These must adhere to a specific format to be valid, but are
straightforward to make. ASLDRO comes with command-line tools to assist creating custom ground truths. For
more information see :doc:`custom_ground_truth`


Built-in ground truths
-----------------------

ASLDRO comes with built-in ground truths:

hrgt_icbm_2009a_nls_3t
~~~~~~~~~~~~~~~~~~~~~~~~~

Normal adult 3T ground truth.
A 1x1x1mm ground truth based on the MNI ICBM 2009a Nonlinear
Symmetric template, obtained from http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009,
with quantitative parameters set based on supplied masks.  This ground truth has the following
values for each tissue and quantity (corresponding to 3T):

+--------------+----------------+--------------+-----------+----------+----------+----------+
| Tissue       | Perfusion Rate | Transit Time | T1        | T2       | T2*      | label    |
|              | [ml/100g/min]  | [s]          | [s]       | [s]      | [s]      |          |
+==============+================+==============+===========+==========+==========+==========+
| Grey Matter  | 60.00          | 0.80         | 1.330     | 0.080    | 0.066    | 1        | 
+--------------+----------------+--------------+-----------+----------+----------+----------+
| White Matter | 20.00          | 1.20         | 0.830     | 0.110    | 0.053    | 2        |
+--------------+----------------+--------------+-----------+----------+----------+----------+
| CSF          | 0.00           | 1000.0       | 3.000     | 0.300    | 0.200    | 3        |
+--------------+----------------+--------------+-----------+----------+----------+----------+

label is the integer value assigned to the tissue in the seg_label volume.


hrgt_icbm_2009a_nls_1.5t
~~~~~~~~~~~~~~~~~~~~~~~~~~

Normal adult 1.5T ground truth.
A 1x1x1mm ground truth based on the MNI ICBM 2009a Nonlinear
Symmetric template, obtained from http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009,
with quantitative parameters set based on supplied masks.  This ground truth has the following
values for each tissue and quantity (corresponding to 1.5T):

+--------------+----------------+--------------+-----------+----------+----------+----------+
| Tissue       | Perfusion Rate | Transit Time | T1        | T2       | T2*      | label    |
|              | [ml/100g/min]  | [s]          | [s]       | [s]      | [s]      |          |
+==============+================+==============+===========+==========+==========+==========+
| Grey Matter  | 60.00          | 0.80         | 1.100     | 0.092    | 0.084    | 1        | 
+--------------+----------------+--------------+-----------+----------+----------+----------+
| White Matter | 20.00          | 1.20         | 0.560     | 0.082    | 0.066    | 2        |
+--------------+----------------+--------------+-----------+----------+----------+----------+
| CSF          | 0.00           | 1000.0       | 3.000     | 0.400    | 0.300    | 3        |
+--------------+----------------+--------------+-----------+----------+----------+----------+

label is the integer value assigned to the tissue in the seg_label volume.

.. _qasper-3t-hrgt:

qasper_3t
~~~~~~~~~~

A ground truth based on the QASPER perfusion phantom (https://goldstandardphantoms.com/qasper),
at 0.5x0.5x0.5mm resolution. Contains the inlet, porous, and outlet regions, and approximates
the behaviour of the QASPER phantom by having spatially variable transit times, and perfusion
rates that are normalised to an input flow rate of 1mL/min. By using the :ref:`ground truth modulate <ground-truth-modulate>`
feature, the scaling factor is then equal to the input flow rate.

+--------+----------------------------------------------------------------+------------------+--------+--------+---------+-----+------------------------+-------+
| Region | Perfusion Rate [ml/100g/min]                                   | Transit Time [s] | T1 [s] | T2 [s] | T2* [s] | M0  | :math:`\lambda` [g/ml] | label |
+--------+----------------------------------------------------------------+------------------+--------+--------+---------+-----+------------------------+-------+
| Inlet  | :math:`100\frac{Q}{N_{\text{inlet}}V\lambda_{\text{inlet}}}`   | 0.0 to 0.25      | 1.80   | 1.20   | 0.90    | 100 | 1.00                   | 1     |
+--------+----------------------------------------------------------------+------------------+--------+--------+---------+-----+------------------------+-------+
| Porous | :math:`100\frac{Q}{N_{\text{porous}}V\lambda_{\text{porous}}}` | 0.25 to 10.0     | 1.80   | 0.20   | 0.10    | 32  | 0.32                   | 2     |
+--------+----------------------------------------------------------------+------------------+--------+--------+---------+-----+------------------------+-------+
| Outlet | :math:`100\frac{Q}{N_{\text{outlet}}V\lambda_{\text{porous}}}` | 10.0 to 20.0     | 1.80   | 1.20   | 0.90    | 100 | 1.00                   | 3     |
+--------+----------------------------------------------------------------+------------------+--------+--------+---------+-----+------------------------+-------+

Q is the bulk flow rate, 1mL/min, N is the number of voxels in a given region, and V is the
voxel volume in mL. Label is the integer value assigned to the tissue in the seg_label volume.

The following single value parameters are also present:

:t1_arterial_blood: 1.8 s
:magnetic_field_strength: 3.0 T

Transit time maps for each region are procedurally generated:

:Inlet: The transit time is proportional to the z position, between 0.0s at z=-20mm and 0.25s
  at z=4.75mm,  which is at the interface between the first and second porous layers.
:Porous: The transit time is generated by subtracting 60 gaussian functions located half way
  along each arteriole at z=4.75mm from a constant value. This is scaled so that the maximum
  value is 10.0s, and the minimum (at the centre of each gaussian) is 0.25s.
:Outlet: The transit time is proportional to the z position, between 10.0s at z=0.0mm, and 20.0s at the
  end of the perfusion chamber at z=45.0mm.

.. note::

    The QASPER ground truth is not an accurate representation of the true QASPER phantom
    behaviour. It cannot be used for comparing experimental QASPER data. It is intended
    for developing and testing image processing pipelines for QASPER data.

    ASL Data generated by ASLDRO for the QASPER ground truth will not show the 'wash-out'
    effect that is seen in real data once the labelled bolus has been delivered and fresh,
    unlabelled perfusate has flowed into the perfusion chamber. This is because the GKM
    implementation in ASLDRO assumes that once the magnetisation has reached the tissue it
    remains there and decays at a rate according to T1. However, in the QASPER phantom there
    is no exchange into tissue and the perfusate keeps moving. The GKM solution needs to
    be extended to incorporate a term for this.