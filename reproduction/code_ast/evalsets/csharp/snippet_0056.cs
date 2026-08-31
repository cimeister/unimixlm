using System;
using System.Collections.Generic;
using Assignment4.Core;
using System.Linq;

namespace Assignment4.Entities
{
    public class TagRepository : ITagRepository
    {
        private KanbanContext _context;

        

        public TagRepository(KanbanContext context)
        {
            _context = context;
        }
        
        public (Response Response, int TagId) Create(TagCreateDTO tag)
        {

            if(_context.Tags.FirstOrDefault(x => x.Name == tag.Name) == null)
            {
                return (Response.Conflict, -1);
            }
            
            var newTag = new Tag{
                Name = tag.Name    
            };

            _context.Tags.Add(newTag);
            _context.SaveChanges();
            
            return (Response.Created, newTag.Id);
               
        }

        public Response Delete(int tagId, bool force = false)
        {
            var tag = _context.Tags.Find(tagId);
            if (tag.tasks.Count > 0 ) 
            {
                if (force)
                {
                    _context.Remove(tag);
                } else 
                {
                    return Response.Conflict;
                }
                
            } else 
            {
                _context.Remove(tag); 
            }
            _context.SaveChanges();
            return Response.Deleted;
        }

        public TagDTO Read(int tagId) //could potentially also return Response.NotFound
        {
            var tag = _context.Tags.Find(tagId);
            return tag != null ? new TagDTO(tag.Id, tag.Name) : null ;
        }

        public IReadOnlyCollection<TagDTO> ReadAll()
        {
            var TagDTOs = new List<TagDTO>(); 
            var tagList = _context.Tags.ToList();
            
            foreach (var tag in tagList)
            {
                var tagDTO = new TagDTO(tag.Id,tag.Name);
                TagDTOs.Add(tagDTO);
            }
            return TagDTOs;
        }

        public Response Update(TagUpdateDTO tag)
        {
            var tagResult = _context.Tags.Find(tag.Id);
            if (tagResult == null){
                return Response.NotFound;
            }
            tagResult.Id = tag.Id;
            tagResult.Name = tag.Name;
 
            _context.SaveChanges();

            return Response.Updated;
        }
    }
}