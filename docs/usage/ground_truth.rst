Input Ground Truth
===================

The input ground truth, known as the High Resolution Ground Truth (or HRGT)
comprises of a 5-dimensional NIFTI image, and json parameter file that
describes the NIFTI and also supplies any non-image ground truth parameters. 3D Volumes for each
ground truth quantity are concatenated along the 5th NIFTI dimension.

.. image:: /images/ground_truth_concatenation.png

Different HRGT's are selected by modifiying the "ground_truth" entry in the input parameter file to
the name of the ground truth being used.  For example:

.. code-block:: json

    {
        "global_configuration": {
            "ground_truth": "hrgt_icbm_2009a_nls_3t"
        },
        ...
    }
will use the built-in ground truth "hrgt_icbm_2009a_nls_3t" (see below for more details of these
datasets). In addition, it is possible to specify your own ground truth files by using one of the
following:

.. code-block:: json

    {
        "global_configuration": {
            "ground_truth": "/path/to/nifti_file.nii"
        },
        ...
    }
or:

.. code-block:: json

    {
        "global_configuration": {
            "ground_truth": {
                "nii": "/path/to/nifti_file.nii.gz"},
                "json": "/path/to/json_file.json"}
        },
        ...
    }
In the two examples above, the first example assumes there is a JSON file at precisely the same path
with the same filename, except for a '.json' extension instead of a '.nii'/'.nii.gz' extension.
The second example uses an explicitly defined filename for each file type, and may have different
paths.

Custom ground truth
~~~~~~~~~~~~~~~~~~~~

To make your own input ground truth please see :doc:`custom_ground_truth`

HRGT parameters and image values can be augmented using the input parameter file. See :doc:`parameters` for more information.
ASLDRO currently supports the following built-in ground truths:


hrgt_icbm_2009a_nls_3t
~~~~~~~~~~~~~~~~~~~~~~

Normal adult 3T ground truth.
A 1x1x1mm ground truth based on the MNI ICBM 2009a Nonlinear
Symmetric template, obtained from http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009,
with quantitative parameters set based on supplied masks.  This ground truth has the following
values for each tissue and quantity (corresponding to 3T):

+--------------+----------------+--------------+-----------+----------+----------+----------+
| Tissue       | Perfusion Rate | Transit Time | T1        | T2       | T2*      | label    |
|              | [ml/100g/min]  | [s]          | [s]       | [s]      | [s]      |          |
+==============+================+==============+===========+==========+==========+==========+
| Grey Matter  | 60.00          | 1.00         | 1.330     | 0.080    | 0.066    | 1        | 
+--------------+----------------+--------------+-----------+----------+----------+----------+
| White Matter | 20.00          | 1.50         | 0.830     | 0.110    | 0.053    | 2        |
+--------------+----------------+--------------+-----------+----------+----------+----------+
| CSF          | 0.00           | 1000.0       | 3.000     | 0.300    | 0.200    | 3        |
+--------------+----------------+--------------+-----------+----------+----------+----------+

label is the integer value assigned to the tissue in the seg_label volume.


hrgt_icbm_2009a_nls_1.5t
~~~~~~~~~~~~~~~~~~~~~~

Normal adult 1.5T ground truth.
A 1x1x1mm ground truth based on the MNI ICBM 2009a Nonlinear
Symmetric template, obtained from http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009,
with quantitative parameters set based on supplied masks.  This ground truth has the following
values for each tissue and quantity (corresponding to 1.5T):

+--------------+----------------+--------------+-----------+----------+----------+----------+
| Tissue       | Perfusion Rate | Transit Time | T1        | T2       | T2*      | label    |
|              | [ml/100g/min]  | [s]          | [s]       | [s]      | [s]      |          |
+==============+================+==============+===========+==========+==========+==========+
| Grey Matter  | 60.00          | 1.00         | 1.100     | 0.092    | 0.084    | 1        | 
+--------------+----------------+--------------+-----------+----------+----------+----------+
| White Matter | 20.00          | 1.50         | 0.560     | 0.082    | 0.066    | 2        |
+--------------+----------------+--------------+-----------+----------+----------+----------+
| CSF          | 0.00           | 1000.0       | 3.000     | 0.400    | 0.300    | 3        |
+--------------+----------------+--------------+-----------+----------+----------+----------+

label is the integer value assigned to the tissue in the seg_label volume.
