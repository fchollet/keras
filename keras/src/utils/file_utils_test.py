import hashlib
import os
import tarfile
import urllib
import urllib.parse
import urllib.request
import zipfile
from unittest.mock import patch

from absl.testing import parameterized

from keras.src.testing import test_case
from keras.src.utils import file_utils
from keras.src.utils.module_utils import LazyModule


class PathToStringTest(test_case.TestCase):
    def test_path_to_string_with_string_path(self):
        path = os.path.join(os.path.sep, "path", "to", "file.txt")
        string_path = file_utils.path_to_string(path)
        self.assertEqual(string_path, path)

    def test_path_to_string_with_PathLike_object(self):
        path = os.path.join(os.path.sep, "path", "to", "file.txt")
        string_path = file_utils.path_to_string(path)
        self.assertEqual(string_path, str(path))

    def test_path_to_string_with_non_string_typed_path_object(self):
        class NonStringTypedPathObject:
            def __fspath__(self):
                return os.path.join(os.path.sep, "path", "to", "file.txt")

        path = NonStringTypedPathObject()
        string_path = file_utils.path_to_string(path)
        self.assertEqual(
            string_path, os.path.join(os.path.sep, "path", "to", "file.txt")
        )

    def test_path_to_string_with_none_path(self):
        string_path = file_utils.path_to_string(None)
        self.assertEqual(string_path, None)


class ResolvePathTest(test_case.TestCase):
    def test_resolve_path_with_absolute_path(self):
        path = os.path.join(os.path.sep, "path", "to", "file.txt")
        resolved_path = file_utils.resolve_path(path)
        self.assertEqual(resolved_path, os.path.realpath(os.path.abspath(path)))

    def test_resolve_path_with_relative_path(self):
        path = os.path.join(".", "file.txt")
        resolved_path = file_utils.resolve_path(path)
        self.assertEqual(resolved_path, os.path.realpath(os.path.abspath(path)))


class ResolveSubPathTest(test_case.TestCase):
    def test_resolve_sub_path_with_absolute_paths(self):
        base_dir = os.path.join(os.path.sep, "path", "to", "base_dir")
        path = os.path.join(base_dir, "file.txt")
        resolved_path = file_utils.resolve_sub_path(base_dir, path)
        self.assertEqual(resolved_path, path)

    def test_resolve_sub_path_with_subdir(self):
        base_dir = os.path.join(os.path.sep, "path", "to", "base_dir")
        path = os.path.join("subdir", "file.txt")
        resolved_path = file_utils.resolve_sub_path(base_dir, path)
        self.assertEqual(
            resolved_path, os.path.realpath(os.path.join(base_dir, path))
        )

    def test_resolve_sub_path_with_same_dir(self):
        base_dir = os.path.join(os.path.sep, "path", "to", "base_dir")
        resolved_path = file_utils.resolve_sub_path(base_dir, ".")
        self.assertEqual(resolved_path, os.path.realpath(base_dir))

    def test_resolve_sub_path_with_path_traversal(self):
        base_dir = os.path.join(os.path.sep, "path", "to", "base_dir")
        path = os.path.join("..", "other_dir", "file.txt")
        self.assertIsNone(file_utils.resolve_sub_path(base_dir, path))

    def test_resolve_sub_path_with_prefix_confusion(self):
        base_dir = os.path.join(os.path.sep, "path", "to", "base_dir")
        path = os.path.join("..", "base_dir_other", "file.txt")
        self.assertIsNone(file_utils.resolve_sub_path(base_dir, path))


class IsLinkInDirTest(test_case.TestCase):
    @parameterized.named_parameters(
        [
            {
                "testcase_name": "hardlink_in",
                "is_symlink": False,
                "name": "file.txt",
                "linkname": "file.txt",
                "expected": True,
            },
            {
                "testcase_name": "hardlink_target_out",
                "is_symlink": False,
                "name": "file.txt",
                "linkname": "../file.txt",
                "expected": False,
            },
            {
                "testcase_name": "hardlink_name_out",
                "is_symlink": False,
                "name": "../file.txt",
                "linkname": "file.txt",
                "expected": False,
            },
            {
                "testcase_name": "symlink_in",
                "is_symlink": True,
                "name": "folder/foo.txt",
                "linkname": "../file.txt",
                "expected": True,
            },
            {
                "testcase_name": "symlink_dest_out",
                "is_symlink": True,
                "name": "../foo.txt",
                "linkname": "file.txt",
                "expected": False,
            },
            {
                "testcase_name": "symlink_link_out",
                "is_symlink": True,
                "name": "folder/foo.txt",
                "linkname": "../../file.txt",
                "expected": False,
            },
        ]
    )
    def test_is_link_in_dir(self, is_symlink, name, linkname, expected):
        base_dir = file_utils.resolve_path(self.get_temp_dir())

        # Creating a TarInfo-like object with a name attribute
        info = type(
            "TarInfo",
            (object,),
            {
                "name": name,
                "linkname": linkname,
                "issym": lambda: is_symlink,
                "islnk": lambda: not is_symlink,
            },
        )

        self.assertEqual(file_utils.is_link_in_dir(info, base_dir), expected)


class FilterSafePathsTest(test_case.TestCase):
    def setUp(self):
        super().setUp()
        self.base_dir = os.path.abspath(self.get_temp_dir())
        self.tar_path = os.path.join(self.base_dir, "test.tar")
        self.target_path = os.path.join(self.base_dir, "target.txt")
        with open(self.target_path, "w") as f:
            f.write("target")
        self.symlink_path = os.path.join(self.base_dir, "symlink.txt")
        os.symlink(self.target_path, self.symlink_path)

    def test_member_within_base_dir(self):
        """Test a member within the base directory."""
        with tarfile.open(self.tar_path, "w") as tar:
            tar.add(self.target_path, arcname="safe_path.txt")
        with tarfile.open(self.tar_path, "r") as tar:
            members = list(
                file_utils.filter_safe_tarinfos(tar.getmembers(), self.base_dir)
            )
            self.assertEqual(len(members), 1)
            self.assertEqual(members[0].name, "safe_path.txt")

    def test_symlink_within_base_dir(self):
        """Test a symlink pointing within the base directory."""
        with tarfile.open(self.tar_path, "w") as tar:
            tar.add(self.symlink_path, arcname="symlink.txt")
        with tarfile.open(self.tar_path, "r") as tar:
            members = list(
                file_utils.filter_safe_tarinfos(tar.getmembers(), self.base_dir)
            )
            self.assertEqual(len(members), 1)
            self.assertEqual(members[0].name, "symlink.txt")

    def test_invalid_path_warning(self):
        """Test warning for an invalid path during archive extraction."""
        with tarfile.open(self.tar_path, "w") as tar:
            tar.add(
                self.target_path, arcname="../../invalid.txt"
            )  # Path intended to be outside of base dir
        with tarfile.open(self.tar_path, "r") as tar:
            with patch("warnings.warn") as mock_warn:
                _ = list(
                    file_utils.filter_safe_tarinfos(
                        tar.getmembers(), self.base_dir
                    )
                )
                warning_msg = (
                    "Skipping invalid path during archive extraction: "
                    "'../../invalid.txt'."
                )
                mock_warn.assert_called_with(warning_msg, stacklevel=2)

    def test_symbolic_link_in_base_dir(self):
        """symbolic link within the base directory is correctly processed."""
        # Add the symbolic link to the tar archive.
        with tarfile.open(self.tar_path, "w") as tar:
            tar.add(self.symlink_path, arcname="symlink.txt")

        with tarfile.open(self.tar_path, "r") as tar:
            members = list(
                file_utils.filter_safe_tarinfos(tar.getmembers(), self.base_dir)
            )
            self.assertEqual(len(members), 1)
            self.assertEqual(members[0].name, "symlink.txt")
            self.assertTrue(
                members[0].issym()
            )  # Explicitly assert it's a symbolic link.

    def test_file_routed_through_symlink_is_rejected(self):
        """A member routed through an in-bounds symlink via `..` is rejected."""
        os.symlink(".", os.path.join(self.base_dir, "link"))
        with tarfile.open(self.tar_path, "w") as tar:
            tar.add(
                self.target_path,
                arcname=os.path.join("link", "..", "escaped.txt"),
            )
        with tarfile.open(self.tar_path, "r") as tar:
            members = list(
                file_utils.filter_safe_tarinfos(tar.getmembers(), self.base_dir)
            )
            self.assertEqual(members, [])


class ExtractArchiveTest(test_case.TestCase):
    def setUp(self):
        super().setUp()
        """Create temporary directories and files for testing."""
        self.temp_dir = self.get_temp_dir()
        self.file_content = "Hello, world!"

        # Create sample files to be archived
        with open(os.path.join(self.temp_dir, "sample.txt"), "w") as f:
            f.write(self.file_content)

    def create_tar(self):
        archive_path = os.path.join(self.temp_dir, "sample.tar")
        with tarfile.open(archive_path, "w") as archive:
            archive.add(
                os.path.join(self.temp_dir, "sample.txt"), arcname="sample.txt"
            )
        return archive_path

    def create_zip(self):
        archive_path = os.path.join(self.temp_dir, "sample.zip")
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.write(
                os.path.join(self.temp_dir, "sample.txt"), arcname="sample.txt"
            )
        return archive_path

    def test_extract_tar(self):
        archive_path = self.create_tar()
        extract_path = os.path.join(self.temp_dir, "extract_tar")
        result = file_utils.extract_archive(archive_path, extract_path, "tar")
        self.assertTrue(result)
        with open(os.path.join(extract_path, "sample.txt"), "r") as f:
            self.assertEqual(f.read(), self.file_content)

    def test_extract_zip(self):
        archive_path = self.create_zip()
        extract_path = os.path.join(self.temp_dir, "extract_zip")
        result = file_utils.extract_archive(archive_path, extract_path, "zip")
        self.assertTrue(result)
        with open(os.path.join(extract_path, "sample.txt"), "r") as f:
            self.assertEqual(f.read(), self.file_content)

    def test_extract_auto(self):
        # This will test the 'auto' functionality
        tar_archive_path = self.create_tar()
        zip_archive_path = self.create_zip()

        extract_tar_path = os.path.join(self.temp_dir, "extract_auto_tar")
        extract_zip_path = os.path.join(self.temp_dir, "extract_auto_zip")

        self.assertTrue(
            file_utils.extract_archive(tar_archive_path, extract_tar_path)
        )
        self.assertTrue(
            file_utils.extract_archive(zip_archive_path, extract_zip_path)
        )

        with open(os.path.join(extract_tar_path, "sample.txt"), "r") as f:
            self.assertEqual(f.read(), self.file_content)

        with open(os.path.join(extract_zip_path, "sample.txt"), "r") as f:
            self.assertEqual(f.read(), self.file_content)

    def test_non_existent_file(self):
        extract_path = os.path.join(self.temp_dir, "non_existent")
        with self.assertRaises(FileNotFoundError):
            file_utils.extract_archive("non_existent.tar", extract_path)

    def test_archive_format_none(self):
        archive_path = self.create_tar()
        extract_path = os.path.join(self.temp_dir, "none_format")
        result = file_utils.extract_archive(archive_path, extract_path, None)
        self.assertFalse(result)

    def test_runtime_error_during_extraction(self):
        tar_path = self.create_tar()
        extract_path = os.path.join(self.temp_dir, "runtime_error_extraction")

        with patch.object(
            tarfile.TarFile, "extractall", side_effect=RuntimeError
        ):
            with self.assertRaises(RuntimeError):
                file_utils.extract_archive(tar_path, extract_path, "tar")
        self.assertFalse(os.path.exists(extract_path))

    def test_keyboard_interrupt_during_extraction(self):
        tar_path = self.create_tar()
        extract_path = os.path.join(
            self.temp_dir, "keyboard_interrupt_extraction"
        )

        with patch.object(
            tarfile.TarFile, "extractall", side_effect=KeyboardInterrupt
        ):
            with self.assertRaises(KeyboardInterrupt):
                file_utils.extract_archive(tar_path, extract_path, "tar")
        self.assertFalse(os.path.exists(extract_path))


class GetFileTest(test_case.TestCase):
    def setUp(self):
        super().setUp()
        """Set up temporary directories and sample files."""
        self.temp_dir = self.get_temp_dir()
        self.file_path = os.path.join(self.temp_dir, "sample_file.txt")
        with open(self.file_path, "w") as f:
            f.write("Sample content")

    def test_valid_tar_extraction(self):
        """Test valid tar.gz extraction and hash validation."""
        dest_dir = self.get_temp_dir()
        orig_dir = self.get_temp_dir()
        _, tar_file_path = self._create_tar_file(orig_dir)
        self._test_file_extraction_and_validation(
            dest_dir, tar_file_path, "tar.gz"
        )

    def test_valid_zip_extraction(self):
        """Test valid zip extraction and hash validation."""
        dest_dir = self.get_temp_dir()
        orig_dir = self.get_temp_dir()
        _, zip_file_path = self._create_zip_file(orig_dir)
        self._test_file_extraction_and_validation(
            dest_dir, zip_file_path, "zip"
        )

    def test_valid_text_file_download(self):
        """Test valid text file download and hash validation."""
        dest_dir = self.get_temp_dir()
        orig_dir = self.get_temp_dir()
        text_file_path = os.path.join(orig_dir, "test.txt")
        with open(text_file_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")
        self._test_file_extraction_and_validation(
            dest_dir, text_file_path, None
        )

    def test_get_file_with_tgz_extension(self):
        """Test extraction of file with .tar.gz extension."""
        dest_dir = self.get_temp_dir()
        orig_dir = dest_dir
        _, tar_file_path = self._create_tar_file(orig_dir)

        origin = urllib.parse.urljoin(
            "file://",
            urllib.request.pathname2url(os.path.abspath(tar_file_path)),
        )

        path = file_utils.get_file(
            "test.txt.tar.gz", origin, untar=True, cache_subdir=dest_dir
        )
        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.exists(os.path.join(path, "test.txt")))

    def test_get_file_with_integrity_check(self):
        """Test file download with integrity check."""
        orig_dir = self.get_temp_dir()
        file_path = os.path.join(orig_dir, "test.txt")

        with open(file_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")

        hashval = file_utils.hash_file(file_path)

        origin = urllib.parse.urljoin(
            "file://", urllib.request.pathname2url(os.path.abspath(file_path))
        )

        path = file_utils.get_file("test.txt", origin, file_hash=hashval)
        self.assertTrue(os.path.exists(path))

    def test_cache_invalidation(self):
        """Test using a hash to force cache invalidation."""
        cache_dir = self.get_temp_dir()
        src_path = os.path.join(self.get_temp_dir(), "test.txt")
        with open(src_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")
        orig_hash = file_utils.hash_file(src_path)
        origin = urllib.parse.urljoin(
            "file://", urllib.request.pathname2url(os.path.abspath(src_path))
        )
        # Download into the cache.
        dest_path = file_utils.get_file(
            "test.txt", origin, file_hash=orig_hash, cache_dir=cache_dir
        )
        self.assertEqual(orig_hash, file_utils.hash_file(dest_path))

        with open(src_path, "w") as text_file:
            text_file.write("Float like a zeppelin, sting like a jellyfish.")
        new_hash = file_utils.hash_file(src_path)
        # Without a hash, we should get the cached version.
        dest_path = file_utils.get_file("test.txt", origin, cache_dir=cache_dir)
        self.assertEqual(orig_hash, file_utils.hash_file(dest_path))
        # Without the new hash, we should re-download.
        dest_path = file_utils.get_file(
            "test.txt", origin, file_hash=new_hash, cache_dir=cache_dir
        )
        self.assertEqual(new_hash, file_utils.hash_file(dest_path))

    def test_force_download(self):
        """Test using a hash to force cache invalidation."""
        cache_dir = self.get_temp_dir()
        src_path = os.path.join(self.get_temp_dir(), "test.txt")
        with open(src_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")
        orig_hash = file_utils.hash_file(src_path)
        origin = urllib.parse.urljoin(
            "file://", urllib.request.pathname2url(os.path.abspath(src_path))
        )
        # Download into the cache.
        dest_path = file_utils.get_file("test.txt", origin, cache_dir=cache_dir)
        self.assertEqual(orig_hash, file_utils.hash_file(dest_path))

        with open(src_path, "w") as text_file:
            text_file.write("Float like a zeppelin, sting like a jellyfish.")
        new_hash = file_utils.hash_file(src_path)
        # Get cached version.
        dest_path = file_utils.get_file("test.txt", origin, cache_dir=cache_dir)
        self.assertEqual(orig_hash, file_utils.hash_file(dest_path))
        # Force download.
        dest_path = file_utils.get_file(
            "test.txt", origin, force_download=True, cache_dir=cache_dir
        )
        self.assertEqual(new_hash, file_utils.hash_file(dest_path))

    def test_get_file_with_failed_integrity_check(self):
        """Test file download with failed integrity check."""
        orig_dir = self.get_temp_dir()
        file_path = os.path.join(orig_dir, "test.txt")

        with open(file_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")

        hashval = "0" * 64

        origin = urllib.parse.urljoin(
            "file://", urllib.request.pathname2url(os.path.abspath(file_path))
        )

        with self.assertRaisesRegex(
            ValueError, "Incomplete or corrupted file.*"
        ):
            _ = file_utils.get_file("test.txt", origin, file_hash=hashval)

    def test_get_file_with_unusable_origin_file_name(self):
        """Test that an origin with no usable file name is rejected."""
        cache_dir = self.get_temp_dir()
        base = urllib.parse.urljoin(
            "file://", urllib.request.pathname2url(os.path.abspath(cache_dir))
        )

        for origin in (f"{base}/", f"{base}/.", f"{base}/.."):
            with self.assertRaisesRegex(
                ValueError, "Can't parse the file name from the origin"
            ):
                _ = file_utils.get_file(origin=origin, cache_dir=cache_dir)

    def _create_tar_file(self, directory):
        """Helper function to create a tar file."""
        text_file_path = os.path.join(directory, "test.txt")
        tar_file_path = os.path.join(directory, "test.tar.gz")
        with open(text_file_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")

        with tarfile.open(tar_file_path, "w:gz") as tar_file:
            tar_file.add(text_file_path, arcname="test.txt")

        return text_file_path, tar_file_path

    def _create_zip_file(self, directory):
        """Helper function to create a zip file."""
        text_file_path = os.path.join(directory, "test.txt")
        zip_file_path = os.path.join(directory, "test.zip")
        with open(text_file_path, "w") as text_file:
            text_file.write("Float like a butterfly, sting like a bee.")

        with zipfile.ZipFile(zip_file_path, "w") as zip_file:
            zip_file.write(text_file_path, arcname="test.txt")

        return text_file_path, zip_file_path

    def _test_file_extraction_and_validation(
        self, dest_dir, file_path, archive_type
    ):
        """Helper function for file extraction and validation."""
        origin = urllib.parse.urljoin(
            "file://",
            urllib.request.pathname2url(os.path.abspath(file_path)),
        )

        hashval_md5 = file_utils.hash_file(file_path, algorithm="md5")

        extract = bool(archive_type)

        path = file_utils.get_file(
            "test",
            origin,
            md5_hash=hashval_md5,
            extract=extract,
            cache_subdir=dest_dir,
        )
        if extract:
            fpath = f"{path}_archive"
        else:
            fpath = path

        self.assertTrue(os.path.exists(path))
        self.assertTrue(file_utils.validate_file(fpath, hashval_md5))
        if extract:
            self.assertTrue(os.path.exists(os.path.join(path, "test.txt")))

    def test_exists(self):
        temp_dir = self.get_temp_dir()
        file_path = os.path.join(temp_dir, "test_exists.txt")

        with open(file_path, "w") as f:
            f.write("test")

        self.assertTrue(file_utils.exists(file_path))
        self.assertFalse(
            file_utils.exists(os.path.join(temp_dir, "non_existent.txt"))
        )

    def test_file_open_read(self):
        temp_dir = self.get_temp_dir()
        file_path = os.path.join(temp_dir, "test_file.txt")
        content = "test content"

        with open(file_path, "w") as f:
            f.write(content)

        with file_utils.File(file_path, "r") as f:
            self.assertEqual(f.read(), content)

    def test_file_open_write(self):
        temp_dir = self.get_temp_dir()
        file_path = os.path.join(temp_dir, "test_file_write.txt")
        content = "test write content"

        with file_utils.File(file_path, "w") as f:
            f.write(content)

        with open(file_path, "r") as f:
            self.assertEqual(f.read(), content)

    def test_isdir(self):
        temp_dir = self.get_temp_dir()
        self.assertTrue(file_utils.isdir(temp_dir))

        file_path = os.path.join(temp_dir, "test_isdir.txt")
        with open(file_path, "w") as f:
            f.write("test")
        self.assertFalse(file_utils.isdir(file_path))

    def test_join_simple(self):
        self.assertEqual(file_utils.join("/path", "to", "dir"), "/path/to/dir")

    def test_join_single_directory(self):
        self.assertEqual(file_utils.join("/path"), "/path")

    def test_listdir(self):
        content = file_utils.listdir(self.temp_dir)
        self.assertIn("sample_file.txt", content)

    def test_walk(self):
        entries = list(file_utils.walk(self.temp_dir))
        self.assertEqual(entries, [(self.temp_dir, [], ["sample_file.txt"])])

    def test_makedirs_and_rmtree(self):
        new_dir = os.path.join(self.temp_dir, "new_directory")
        file_utils.makedirs(new_dir)
        self.assertTrue(os.path.isdir(new_dir))
        file_utils.rmtree(new_dir)
        self.assertFalse(os.path.exists(new_dir))

    def test_copy(self):
        dest_path = os.path.join(self.temp_dir, "copy_sample_file.txt")
        file_utils.copy(self.file_path, dest_path)
        self.assertTrue(os.path.exists(dest_path))
        with open(dest_path, "r") as f:
            content = f.read()
        self.assertEqual(content, "Sample content")

    def test_remove_sub_directory(self):
        parent_dir = os.path.join(self.get_temp_dir(), "parent_directory")
        child_dir = os.path.join(parent_dir, "child_directory")
        file_utils.makedirs(child_dir)
        file_utils.rmtree(parent_dir)
        self.assertFalse(os.path.exists(parent_dir))
        self.assertFalse(os.path.exists(child_dir))

    def test_remove_files_inside_directory(self):
        dir_path = os.path.join(self.get_temp_dir(), "test_directory")
        file_path = os.path.join(dir_path, "test.txt")
        file_utils.makedirs(dir_path)
        with open(file_path, "w") as f:
            f.write("Test content")
        file_utils.rmtree(dir_path)
        self.assertFalse(os.path.exists(dir_path))
        self.assertFalse(os.path.exists(file_path))

    def test_handle_complex_paths(self):
        complex_dir = os.path.join(self.get_temp_dir(), "complex dir@#%&!")
        file_utils.makedirs(complex_dir)
        file_utils.rmtree(complex_dir)
        self.assertFalse(os.path.exists(complex_dir))


class HashFileTest(test_case.TestCase):
    def setUp(self):
        super().setUp()
        self.test_content = b"Hello, World!"
        self.temp_file = os.path.join(self.get_temp_dir(), "test_file.txt")
        with open(self.temp_file, "wb") as f:
            f.write(self.test_content)

    def test_hash_file_sha256(self):
        """Test SHA256 hashing of a file."""
        expected_sha256 = (
            "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        )
        calculated_sha256 = file_utils.hash_file(
            self.temp_file, algorithm="sha256"
        )
        self.assertEqual(expected_sha256, calculated_sha256)

    def test_hash_file_md5(self):
        """Test MD5 hashing of a file."""
        expected_md5 = "65a8e27d8879283831b664bd8b7f0ad4"
        calculated_md5 = file_utils.hash_file(self.temp_file, algorithm="md5")
        self.assertEqual(expected_md5, calculated_md5)


class TestValidateFile(test_case.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_file = os.path.join(self.get_temp_dir(), "test_file.txt")
        with open(self.temp_file, "wb") as f:
            f.write(b"Hello, World!")

        self.sha256_hash = (
            "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        )
        self.md5_hash = "65a8e27d8879283831b664bd8b7f0ad4"

    def test_validate_file_sha256(self):
        """Validate SHA256 hash of a file."""
        self.assertTrue(
            file_utils.validate_file(self.temp_file, self.sha256_hash, "sha256")
        )

    def test_validate_file_md5(self):
        """Validate MD5 hash of a file."""
        self.assertTrue(
            file_utils.validate_file(self.temp_file, self.md5_hash, "md5")
        )

    def test_validate_file_auto_sha256(self):
        """Auto-detect and validate SHA256 hash."""
        self.assertTrue(
            file_utils.validate_file(self.temp_file, self.sha256_hash, "auto")
        )

    def test_validate_file_auto_md5(self):
        """Auto-detect and validate MD5 hash."""
        self.assertTrue(
            file_utils.validate_file(self.temp_file, self.md5_hash, "auto")
        )

    def test_validate_file_wrong_hash(self):
        """Test validation with incorrect hash."""
        wrong_hash = "deadbeef" * 8
        self.assertFalse(
            file_utils.validate_file(self.temp_file, wrong_hash, "sha256")
        )


class ResolveHasherTest(test_case.TestCase):
    def test_resolve_hasher_sha256(self):
        """Test resolving hasher for sha256 algorithm."""
        hasher = file_utils.resolve_hasher("sha256")
        self.assertIsInstance(hasher, type(hashlib.sha256()))

    def test_resolve_hasher_auto_sha256(self):
        """Auto-detect and resolve hasher for sha256."""
        hasher = file_utils.resolve_hasher("auto", file_hash="a" * 64)
        self.assertIsInstance(hasher, type(hashlib.sha256()))

    def test_resolve_hasher_auto_md5(self):
        """Auto-detect and resolve hasher for md5."""
        hasher = file_utils.resolve_hasher("auto", file_hash="a" * 32)
        self.assertIsInstance(hasher, type(hashlib.md5()))

    def test_resolve_hasher_default(self):
        """Resolve hasher with a random algorithm value."""
        hasher = file_utils.resolve_hasher("random_value")
        self.assertIsInstance(hasher, type(hashlib.md5()))


class IsRemotePathTest(test_case.TestCase):
    def test_gcs_remote_path(self):
        self.assertTrue(file_utils.is_remote_path("/gcs/some/path/to/file.txt"))
        self.assertTrue(file_utils.is_remote_path("/gcs/another/directory/"))
        self.assertTrue(file_utils.is_remote_path("gcs://bucket/some/file.txt"))

    def test_hdfs_remote_path(self):
        self.assertTrue(file_utils.is_remote_path("hdfs://some/path/on/hdfs"))
        self.assertTrue(file_utils.is_remote_path("/hdfs/some/local/path"))

    def test_cns_remote_path(self):
        self.assertTrue(file_utils.is_remote_path("/cns/some/path"))

    def test_placer_remote_path(self):
        self.assertTrue(
            file_utils.is_remote_path("/placer/prod/home/some/path")
        )
        self.assertTrue(
            file_utils.is_remote_path("/placer/test/home/some/path")
        )
        self.assertTrue(
            file_utils.is_remote_path("/placer/prod/scratch/home/some/path")
        )

    def test_tfhub_remote_path(self):
        self.assertTrue(file_utils.is_remote_path("/tfhub/some/path"))

    def test_cfs_remote_path(self):
        self.assertTrue(file_utils.is_remote_path("/cfs/some/path"))

    def test_readahead_remote_path(self):
        self.assertTrue(file_utils.is_remote_path("/readahead/some/path"))

    def test_namespace_remote_path(self):
        self.assertTrue(file_utils.is_remote_path("/namespace/some/path"))

    def test_non_remote_paths(self):
        self.assertFalse(file_utils.is_remote_path("/local/path/to/file.txt"))
        self.assertFalse(
            file_utils.is_remote_path("C:\\local\\path\\on\\windows\\file.txt")
        )
        self.assertFalse(file_utils.is_remote_path("~/relative/path/"))
        self.assertFalse(file_utils.is_remote_path("./another/relative/path"))
        self.assertFalse(file_utils.is_remote_path("/local/path"))
        self.assertFalse(file_utils.is_remote_path("./relative/path"))
        self.assertFalse(file_utils.is_remote_path("~/relative/path"))


class RemotePathTest(test_case.TestCase):
    # A module that can never be imported, used to hide `gfile` / `epath`.
    _UNAVAILABLE = LazyModule("keras_test_unavailable_module")

    def test_is_remote_path_or_raise_raises_correct_message(self):
        path = "gs://bucket/some/file.txt"
        expected_error_msg = (
            r"Handling remote paths requires installing etils\[epath\] "
            r"\(e\.g\. `pip install keras\[gcs\]`\) or TensorFlow\. "
            f"Received: path={path}"
        )
        with (
            patch.object(file_utils, "gfile", self._UNAVAILABLE),
            patch.object(file_utils, "epath", self._UNAVAILABLE),
        ):
            with self.assertRaisesRegex(ValueError, expected_error_msg):
                file_utils._is_remote_path_or_raise(path)
            self.assertFalse(
                file_utils._is_remote_path_or_raise("/local/path/file.txt")
            )

    def test_epath_remote_path(self):
        root = self.get_temp_dir()
        # Send local paths through the remote branch, and hide TensorFlow so
        # that `epath` is used.
        with (
            patch.object(file_utils, "is_remote_path", lambda p: True),
            patch.object(file_utils, "gfile", self._UNAVAILABLE),
        ):
            subdir = file_utils.join(root, "a", "b")
            self.assertEqual(subdir, os.path.join(root, "a", "b"))
            self.assertEqual(file_utils.join(subdir, ""), subdir + "/")
            self.assertEqual(file_utils.join(subdir + "/"), subdir + "/")
            file_utils.makedirs(subdir)
            file_utils.makedirs(subdir)  # Tolerates an existing directory.
            self.assertTrue(file_utils.isdir(subdir))

            src = file_utils.join(subdir, "src.txt")
            dst = file_utils.join(subdir, "dst.txt")
            with file_utils.File(src, "w") as f:
                f.write("new")
            with file_utils.File(dst, "w") as f:
                f.write("old")
            file_utils.copy(src, dst)  # Overwrites.
            with file_utils.File(dst, "r") as f:
                self.assertEqual(f.read(), "new")
            self.assertEqual(
                sorted(file_utils.listdir(subdir)), ["dst.txt", "src.txt"]
            )
            parent = file_utils.join(root, "a")
            self.assertEqual(
                [
                    (dirpath, dirs, sorted(files))
                    for dirpath, dirs, files in file_utils.walk(parent)
                ],
                [
                    (parent, ["b"], []),
                    (subdir, [], ["dst.txt", "src.txt"]),
                ],
            )

            file_utils.remove(dst)
            self.assertFalse(file_utils.exists(dst))
            file_utils.rmtree(parent)
            self.assertFalse(file_utils.exists(subdir))
