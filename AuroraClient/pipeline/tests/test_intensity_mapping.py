"""Per-edit intensity reverse maps, including the legacy single-dict form."""

import unittest

from pipeline.intensityMapping import (
    compose_intensity_mapping,
    drop_intensity_mappings_containing,
    record_intensity_value_mapping,
    remove_intensity_mappings_for_filenames,
    rename_intensity_mapping_filenames,
    resolve_intensity_value_mapping,
)


BASE = {
    'mode': 'shared_project',
    'mapping_scale': 2.0,
    'mapping_shift': 10.0,
    'window_min': 10.0,
    'window_max': 20.0,
}


class IntensityMappingTests(unittest.TestCase):
    def test_compose_offset_and_scale(self):
        # stored_new = stored_old - 5
        shifted = compose_intensity_mapping(BASE, stored_scale=1.0, stored_offset=-5.0)
        self.assertEqual(shifted['mapping_scale'], 2.0)
        self.assertEqual(shifted['mapping_shift'], 20.0)
        self.assertEqual(shifted['window_min'], 10.0)

        # stored_new = 4 * stored_old + 1
        stretched = compose_intensity_mapping(BASE, stored_scale=4.0, stored_offset=1.0)
        self.assertEqual(stretched['mapping_scale'], 0.5)
        self.assertEqual(stretched['mapping_shift'], 9.5)

    def test_legacy_dict_applies_to_every_edit(self):
        self.assertIs(
            resolve_intensity_value_mapping(BASE, 'subject_lossy_edit_2_backfixed.nii.gz'),
            BASE,
        )
        self.assertIs(resolve_intensity_value_mapping(BASE, None), BASE)

    def test_record_promotes_legacy_dict_and_delete_removes_only_that_edit(self):
        metadata = {'name': 'subject', 'intensity_value_mapping': dict(BASE)}
        self.assertTrue(record_intensity_value_mapping(
            metadata,
            'subject_lossy_edit_0_backfixed.nii.gz',
            stored_offset=-5.0,
        ))
        rows = metadata['intensity_value_mapping']
        self.assertEqual(
            [row['filename'] for row in rows],
            [
                'subject_lossy.nii.gz',
                'subject_lossy_edit_0_backfixed.nii.gz',
            ],
        )
        self.assertEqual(rows[0]['mapping_shift'], 10.0)
        self.assertEqual(rows[1]['mapping_shift'], 20.0)

        # An edit saved before the list existed still resolves to the raw pair.
        legacy_edit = resolve_intensity_value_mapping(rows, 'subject_lossy_edit_9_cropped.nii.gz')
        self.assertEqual(legacy_edit['mapping_shift'], 10.0)
        backfixed = resolve_intensity_value_mapping(rows, 'subject_lossy_edit_0_backfixed.nii.gz')
        self.assertEqual(backfixed['mapping_shift'], 20.0)

        self.assertTrue(record_intensity_value_mapping(
            metadata,
            'subject_lossy_edit_1_cropped.nii.gz',
        ))
        cropped = resolve_intensity_value_mapping(
            metadata['intensity_value_mapping'],
            'subject_lossy_edit_1_cropped.nii.gz',
        )
        self.assertEqual(cropped['mapping_shift'], 20.0)

        self.assertTrue(remove_intensity_mappings_for_filenames(
            metadata,
            ['subject_lossy_edit_0_backfixed.nii.gz'],
        ))
        names = [row['filename'] for row in metadata['intensity_value_mapping']]
        self.assertNotIn('subject_lossy_edit_0_backfixed.nii.gz', names)
        self.assertIn('subject_lossy_edit_1_cropped.nii.gz', names)

        legacy = {'intensity_value_mapping': dict(BASE)}
        self.assertFalse(remove_intensity_mappings_for_filenames(
            legacy,
            ['subject_lossy_edit_0_backfixed.nii.gz'],
        ))
        self.assertEqual(legacy['intensity_value_mapping'], BASE)

    def test_recording_the_raw_file_replaces_the_list(self):
        metadata = {
            'name': 'atlas',
            'intensity_value_mapping': [
                {'filename': 'reference_lossy.nii.gz', 'mapping_scale': 2.0, 'mapping_shift': 10.0},
                {'filename': 'reference_lossy_edit_1_cropped.nii.gz', 'mapping_scale': 2.0, 'mapping_shift': 10.0},
            ],
        }
        self.assertTrue(record_intensity_value_mapping(metadata, 'atlas_lossy.nii.gz'))
        rows = metadata['intensity_value_mapping']
        self.assertEqual([row['filename'] for row in rows], ['atlas_lossy.nii.gz'])
        self.assertEqual(rows[0]['mapping_shift'], 10.0)

    def test_rename_and_drop_ignore_legacy_dict(self):
        metadata = {'intensity_value_mapping': dict(BASE)}
        self.assertFalse(rename_intensity_mapping_filenames(metadata, '_edit_1_elastic', '_edit_0_elastic'))
        self.assertFalse(drop_intensity_mappings_containing(metadata, '_masked'))

        listed = {
            'intensity_value_mapping': [
                {'filename': 'subject_lossy.nii.gz', 'mapping_scale': 1.0, 'mapping_shift': 0.0},
                {'filename': 'subject_lossy_edit_1_elastic.nii.gz', 'mapping_scale': 1.0, 'mapping_shift': 4.0},
                {'filename': 'subject_lossy_edit_1_masked.nii.gz', 'mapping_scale': 1.0, 'mapping_shift': 4.0},
            ]
        }
        self.assertTrue(drop_intensity_mappings_containing(listed, '_edit_1_masked'))
        self.assertTrue(rename_intensity_mapping_filenames(listed, '_edit_1_elastic', '_edit_0_elastic'))
        names = [row['filename'] for row in listed['intensity_value_mapping']]
        self.assertEqual(names, [
            'subject_lossy.nii.gz',
            'subject_lossy_edit_0_elastic.nii.gz',
        ])
