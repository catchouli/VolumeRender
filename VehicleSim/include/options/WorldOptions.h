#ifndef VEHICLESIM_WORLDOPTIONS
#define VEHICLESIM_WORLDOPTIONS

#include <Box2D/Box2D.h>

namespace vlr
{
	class WorldOptions
	{
	public:
		WorldOptions()
			: _world(nullptr)
		{

		}

		WorldOptions(b2World* world, b2Draw* draw)
			: _world(world), _draw(draw)
		{
		
		}

		void setWorld(b2World* world)
		{
			_world = world;
		}

		float getGravityX() const
		{
			return _world->GetGravity().x;
		}

		void setGravityX(float val)
		{
			b2Vec2 gravity(val, _world->GetGravity().y);

			_world->SetGravity(gravity);
		}

		float getGravityY() const
		{
			return _world->GetGravity().y;
		}

		void setGravityY(float val)
		{
			b2Vec2 gravity(_world->GetGravity().x, val);

			_world->SetGravity(gravity);
		}

		void setEnableDrawShapes(bool val)
		{
			uint32 flags = _draw->GetFlags();

			if (val)
				flags |= b2Draw::e_shapeBit;
			else
				flags &= ~b2Draw::e_shapeBit;

			_draw->SetFlags(flags);
		}

		bool getEnableDrawShapes() const
		{
			uint32 flags = _draw->GetFlags();
			
			return (flags & b2Draw::e_shapeBit) > 0;
		}

		void setEnableDrawJoints(bool val)
		{
			uint32 flags = _draw->GetFlags();

			if (val)
				flags |= b2Draw::e_jointBit;
			else
				flags &= ~b2Draw::e_jointBit;

			_draw->SetFlags(flags);
		}

		bool getEnableDrawJoints() const
		{
			uint32 flags = _draw->GetFlags();
			
			return (flags & b2Draw::e_jointBit) > 0;
		}

		void setEnableDrawAABBs(bool val)
		{
			uint32 flags = _draw->GetFlags();

			if (val)
				flags |= b2Draw::e_aabbBit;
			else
				flags &= ~b2Draw::e_aabbBit;

			_draw->SetFlags(flags);
		}

		bool getEnableDrawAABBs() const
		{
			uint32 flags = _draw->GetFlags();
			
			return (flags & b2Draw::e_aabbBit) > 0;
		}

	private:
		b2World* _world;
		b2Draw* _draw;
	};
}

#endif /* VEHICLESIM_WORLDOPTIONS */
