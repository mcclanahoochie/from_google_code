--
-- Quick Export Script to Pyr Detail
--   by Chris McClanahan http://mcclanahoochie.com/blog/
--   http://mcclanahoochie.com/blog/2013/01/lightroom-plugins-infrared-and-pyrdetail
--

local LrTasks = import 'LrTasks'
local LrApplication = import 'LrApplication'
local LrLogger = import 'LrLogger'

local myLogger = LrLogger('PyrDetail')
myLogger:enable("print")

exportServiceProvider = {}
exportServiceProvider.canExportVideo = false

function exportServiceProvider.processRenderedPhotos(functionContext, exportContext)
   
   local exportSession = exportContext.exportSession
   local exportSettings = exportContext.propertyTable
   local nPhotos = exportSession:countRenditions()
   local progressScope = exportContext:configureProgress {
      title =
         nPhotos > 1 and LOC("$$$/PyrDetail/Publish/Progress=Exporting ^1 photos to Pyr Detail", nPhotos)
         or              LOC "$$$/PyrDetail/Publish/Progress/One=Exporting one photo to Pyr Detail",
   }
   
   for i, rendition in exportContext:renditions { stopIfCanceled = true } do		
      progressScope:setPortionComplete((i - 1) / nPhotos)
      if not rendition.wasSkipped then
         local success, pathOrMessage = rendition:waitForRender()
         progressScope:setPortionComplete((i - 0.5) / nPhotos)
         if progressScope:isCanceled() then break end
         if success then
            local filePath = assert(pathOrMessage)
            -- configure bin paths --
            -- result = LrTasks.execute("/Applications/pyr_detail.app/Contents/MacOS/pyr_detail " .. filePath)
            -- result = LrTasks.execute("/Users/chris/workspace/bin_lapl/pyr_detail.app/Contents/MacOS/pyr_detail " .. filePath)
            result = LrTasks.execute("/Users/chris/workspace/pyrdetail_bitbucket/pyr_detail.app/Contents/MacOS/pyr_detail " .. filePath)
            -- reimport
            if result == 0 then
               local catalog = LrApplication:activeCatalog()
               catalog:withWriteAccessDo('Import from PyrDetail',
                                         function(context) 
                                            catalog:addPhoto(filePath)
                                         end
                                        )
            end
         end
      end
   end

   progressScope:done()

end

return exportServiceProvider
