--
-- Quick Export Script to Pyr Detail
--   by Chris McClanahan http://mcclanahoochie.com/blog/
--

return {
	LrSdkVersion = 4.0,
	LrSdkMinimumVersion = 3.0,
	LrToolkitIdentifier = 'com.adobe.lightroom.export.pyrdetail',
	LrPluginName = LOC "$$$/YAPB/PluginName=Pyr Detail",
	LrExportServiceProvider = {
		title = LOC "$$$/YAPB/YAPB-title=PyrDetail",
		file = 'PyrDetailExportServiceProvider.lua',
                builtInPresetsDir = 'presets',
	},
	VERSION = {
           major=1, minor=0, revision=0, build=1,
        },
}
