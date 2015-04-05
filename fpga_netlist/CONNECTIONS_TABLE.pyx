#distutils: language=c++
#cython: embedsignature=True, boundscheck=False
from libc.stdint cimport int32_t, uint8_t
from cython.operator cimport dereference as deref
from cythrust.thrust.unique cimport counted_unique
from cythrust.thrust.copy cimport copy
from cythrust.thrust.functional cimport (multiplies, plus,
                                         unpack_binary_args)
from cythrust.thrust.sort cimport sort
from cythrust.thrust.iterator.constant_iterator cimport \
    make_constant_iterator
from cythrust.thrust.iterator.transform_iterator cimport \
    make_transform_iterator
from cythrust.thrust.transform cimport transform2
from cythrust.thrust.tuple cimport make_tuple2
from cythrust.thrust.iterator.permutation_iterator cimport \
    make_permutation_iterator
from cythrust.thrust.iterator.zip_iterator cimport make_zip_iterator
from cythrust.device_vector cimport (DeviceVectorViewInt32,
                                     DeviceVectorViewUint8)
from cythrust.thrust.device_vector cimport device_vector


def block_type_by_block_key(DeviceVectorViewInt32 block_key,
                            DeviceVectorViewUint8 block_type):
    sort(make_zip_iterator(make_tuple2(block_key._begin,
                                       block_type._begin)),
         make_zip_iterator(make_tuple2(block_key._end,
                                       block_type._end)))

    cdef size_t count = counted_unique(
         make_zip_iterator(make_tuple2(block_key._begin,
                                       block_type._begin)),
         make_zip_iterator(make_tuple2(block_key._end,
                                       block_type._end)))
    return count


def driver_and_sink_type(DeviceVectorViewInt32 driver_key,
                         DeviceVectorViewInt32 sink_key,
                         DeviceVectorViewUint8 driver_type,
                         DeviceVectorViewUint8 sink_type,
                         DeviceVectorViewUint8 block_types):
    copy(
        make_zip_iterator(
            make_tuple2(
                make_permutation_iterator(block_types._begin,
                                          driver_key._begin),
                make_permutation_iterator(block_types._begin,
                                          sink_key._begin))),
        make_zip_iterator(
            make_tuple2(
                make_permutation_iterator(block_types._begin,
                                          driver_key._end),
                make_permutation_iterator(block_types._begin,
                                          sink_key._end))),
        make_zip_iterator(
            make_tuple2(driver_type._begin, sink_type._begin)))


def connection_delay_type(DeviceVectorViewUint8 driver_type,
                          DeviceVectorViewUint8 sink_type,
                          DeviceVectorViewUint8 delay_type):
    cdef multiplies[uint8_t] multiply_f
    cdef plus[uint8_t] plus_f
    cdef unpack_binary_args[multiplies[uint8_t]] *wrapped_multiply_f = \
        new unpack_binary_args[multiplies[uint8_t]](multiply_f)

    transform2(
        driver_type._begin, driver_type._end,
        make_transform_iterator(
            make_zip_iterator(make_tuple2(make_constant_iterator(10),
                                          sink_type._begin)),
            deref(wrapped_multiply_f)), delay_type._begin, plus_f)
